from httplib2 import Http
from json import dumps
from chatty import chat

def main():
    url = ""
    bot_message = {
        'text': 'Hello! I\'m Chatty, nice to meet you! '
    }

    message_headers = { 'Content-Type': 'application/json; charset = UTF-8'}

    http_obj = Http()

    response = http_obj.request(
        url = url,
        method = 'POST',
        headers = message_headers,
        body = dumps(bot_message)
    )

    print(response)

if __name__ == '__main__':
    main()