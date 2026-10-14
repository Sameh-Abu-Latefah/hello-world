from hello_world.main import hello


def test_hello():
    assert hello() == "Hello, World!"
    print("\n"+__name__)