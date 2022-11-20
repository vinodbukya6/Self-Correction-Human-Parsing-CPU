
def success_response(code,message,response,success):
    
    result = {
        "code": code,
        'message': message,
        "response": response,
        "success": success
        }

    return result 


def failed_response(code,message):
    
    result = {
        "code": code,
        'message': message,
        "response": {},
        "success": False
        }

    return result
