import os
import re
import io
import time
import json
import signal
import contextlib
import subprocess
import func_timeout

from LynLLM import cli_api
from LynLLM.config.run_config import Configuration


class BaseLynllm:
    def __init__(self, 
                 model_path, 
                 device_list, 
                 **kwargs):
        self.model_path = model_path
        self.device_list = device_list
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.other_args = ' '.join([f"--{key}" if value == True else "" if isinstance(value, bool) \
                               else f"--{key} {value}" for key, value in kwargs.items()])
        self.start()
    
    def start(self):
        NotImplementedError
        
    def gererate(self, query, **kwargs):
        NotImplementedError
        
    def terminate(self):
         NotImplementedError
         
    def __del__(self):
        self.terminate()
    
    @staticmethod   
    def extract_speed(text):
        pattern = r'\|(\s*[^|]+\s*)\|(\s*[\d.]+\s*)\|'
        matches = re.findall(pattern, text)
        result = {key.strip(): float(value.strip()) for key, value in matches}
        return result


class CommandLine(BaseLynllm):
    def __init__(self, model_path, device_list, **kwargs):
        super().__init__(model_path, device_list, **kwargs)
        self.end_marker = "输入内容可进行对话，clear 清空对话历史，stop 终止程序"
        
    def start(self):
        if hasattr(self, "process"):
            if self.process and self.process.poll() is None:
                return
        
        cmd = f"lynllm-cli-app -m {self.model_path} -d {self.device_list} {self.other_args}"
        print(cmd)
        self.process = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid
        )   
        
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("lynllm-cli-app not running.")
    
    def generate(self, query, **kwargs):        
        if not self.process or self.process.poll() is not None:
            print("lynllm-cli-app not running.")
            return
        
        query = query.strip()
        
        self.process.stdin.write(query+'\n')
        self.process.stdin.flush()
        
        if query == "clear" or query == "stop":
            return
        
        output = ""
        error = ""
        
        while True:
            if self.process.poll() is not None:
                err = self.process.stderr.readline()
                while len(err):
                    error += err
                    err = self.process.stderr.readline()
                    
                if len(error):
                    self.terminate()
                    raise RuntimeError(error)
                
                break
                
            out = self.process.stdout.readline()
               
            if self.end_marker in out:
                break
            
            output += out
        
        start = output.find("用户 >") + 4
        end = output.find("+--------------------------------------------------+---------------+")

        content = output[start:end] if end != -1 else output[start:]
        content = content.strip()
                
        text = output[end:] if end != -1 else ''
        result = type(self).extract_speed(text)
        
        result["content"] = content
        
        return result
    
    def terminate(self):
        if self.process and self.process.poll() is None: 
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)  
            self.process.wait()
            self.process = None
            time.sleep(5)
        # self.generate("stop")
        # if self.process and self.process.poll() is None:
        #     self.process.stdin.close()
        #     self.process.stdout.close()
        #     self.process.stderr.close()
        #     self.process.terminate()
        #     self.process.wait()
        #     self.process = None

        
class LLMApi(BaseLynllm):        
    def start(self):
        if hasattr(self, "model"):
            return
        if hasattr(self, "other_args"):
            delattr(self, "other_args")
        if isinstance(self.device_list, str):
            self.device_list = list(map(int, self.device_list.split(',')))
        self.model, self.tokenizer, self.generation_config = cli_api.init_llm_model(**vars(self))
        self.rc9 = Configuration()
    
    def generate(self, query, **kwargs):
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer): 
            content = cli_api.run_model(query, self.model, self.tokenizer, self.generation_config, **kwargs)
            print(content)
            breakpoint()
        output = output_buffer.getvalue()
        result = self.__class__.extract_speed(output)
        # Fix: content is a generator, convert to list or use next()
        if hasattr(content, '__iter__') and not isinstance(content, (str, bytes)):
            try:
                result["content"] = next(content)
            except StopIteration:
                result["content"] = ""
        else:
            result["content"] = content
        return result
    
    def terminate(self):
        if self.model:
            self.rc9.teardown()
            self.model = None
            self.tokenizer = None
            self.generation_config = None
            self.rc9 = None
    
        
class OpenaiService(BaseLynllm):
    @func_timeout.func_set_timeout(60)
    def check_server(self):
        cmd = f"curl http://{self.host}:{self.port}/v1/models"
        while True:
            ret = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ret.returncode == 0:
                output = ret.stdout.decode("utf-8")
                print(output)
                break
            time.sleep(3)
            
        if not self.server or self.server.poll() is not None:
            raise RuntimeError("lynllm-serving not running.")
    
    def start(self):
        if hasattr(self, "server"):
            if self.server and self.server.poll() is None:
                return
        
        cmd = f"lynllm-serving --model {self.model_path} -d {self.device_list} {self.other_args}"
        print(cmd)
        self.server = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)   
        try:
            self.check_server()
        except func_timeout.exceptions.FunctionTimedOut:
            raise RuntimeError("lynllm-serving start timeout.")
            
    def generate(self, query, **kwargs):
        if not self.server or self.server.poll() is not None:
            print("lynllm-serving not running.")
            return 
        
        data = {
            "model": f"{self.model_path}",
            "messages":[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"{query}"
                }
            ],
            "stream": False
        }
        
        cmd = [
            "curl",
            f"http://{self.host}:{self.port}/v1/chat/completions",
            "-H", "Content-Type:application/json",
            "-d", json.dumps(data)
        ]
        print(' '.join(cmd))
        
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if ret.returncode != 0:
            self.terminate()
            error = ret.stderr.decode("utf-8")
            raise RuntimeError(error)
        
        result = {}

        output= ret.stdout.decode("utf-8")
        try:
            output = json.loads(output)
            content = output["choices"][0]["message"]["content"]
        except json.decoder.JSONDecodeError:
            content = output

        result["content"] = content
        
        return result
    
    def terminate(self):
        if self.server and self.server.poll() is None: 
            os.killpg(os.getpgid(self.server.pid), signal.SIGTERM)  
            self.server.wait()
            self.server = None
            time.sleep(5)
    