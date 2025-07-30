import pickle

# SingleTon을 상속받은 class는 initializer만 구현하면 singleton으로 사용 가능.
class BaseSingleTon:
    def __new__(cls, *args, **kwargs):
        # Foo 클래스 객체에 _instance 속성이 없다면
        if not hasattr(cls, "_instance"):
            # print("SimConfig __new__ is called")
            # Foo 클래스의 객체를 생성하고 Foo._instance로 바인딩
            cls._instance = super().__new__(cls)
        return cls._instance  # Foo._instance를 리턴

    def __init__(self):
        cls = type(self)
        # Foo 클래스 객체에 _init 속성이 없다면
        if not hasattr(cls, "_init"):
            # print("SimConfig __init__ is called")
            cls._init = True
            self.init()  # 자식에서 반드시 구현되어야 한다.

    def serialize(self) -> bytes:
        return pickle.dumps(self.__dict__)

    def deserialize(self, state_dict: bytes):
        self.__dict__.update(pickle.loads(state_dict))
