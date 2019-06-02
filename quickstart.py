from httplib2 import Http
from json import dumps
from chatty import chat

def main():
    url = "https://chat.googleapis.com/v1/spaces/AAAAhRq0VFI/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=UsVKw3RU5q8ldkI1_WYWAzrvgDexNeBEVWuQky9s5NE%3D"
    bot_message = {
        'text': 'Hello! I\'m Chatty, nice to meet you! '
    }

    message_headers = { 'Content-Type': 'application/json; charset = UTF-8'}

    http_obj = Http()

    response = http_obj.request(
        uri = url,
        method = 'POST',
        headers = message_headers,
        body = dumps(bot_message)
    )

    print(response)

if __name__ == '__main__':
    main()