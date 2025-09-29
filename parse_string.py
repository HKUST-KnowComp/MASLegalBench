import re
import json

class Parser:
    def __init__(self):
        return

        
    def parse_answer(self, response):
        ret = {
            "response": response.strip()
        }
        try:
            json_content = json.loads(response)
            ret["json"] = json_content
            answer = json_content["answer"]
            ret["decision"] = answer
            return ret
        except:
            pass

        # 搜索{}之间的部分，含有{}
        json_match = re.search(r"\{.*?\}", response, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_content = json.loads(json_match.group(0).strip())
            ret["json"] = json_content
            answer = json_content["answer"]
            ret["decision"] = answer
            return ret

        json_match = re.search(r"```json(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_content = json.loads(json_match.group(1).strip())
            ret["json"] = json_content
            answer = json_content["answer"]
            ret["decision"] = answer
            return ret
        else:
            print(response)
            raise ValueError("No valid JSON found in response.")

