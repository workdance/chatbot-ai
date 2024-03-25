from playground.ollama_function import get_current_weather


def test_get_weather():
    response = get_current_weather(**{
        "location": "北京"
    })

    print(response)