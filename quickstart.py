from httplib2 import Http
from json import dumps

def main():
    url = ""
    bot_message = {
        'text': 'Hello! I\'m Chatty, nice to meet you! '
    }

    message_headers = { 'Content-Type': 'application/json; charset = UTF-8'}

    http_obj = Http()