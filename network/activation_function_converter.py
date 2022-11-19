class ActivationFunctionConverter():

    def __init__(self) -> None:
        self.__id_to_name = {
            1: 'Linear',
            2: 'Step',
            3: 'Sin',
            4: 'Gausian',
            5: 'Tanh',
            6: 'Sigmoid',
            7: 'Inverse',
            8: 'Absolute',
            9: 'Relu',
            10: 'Cosine',
            11: 'Squared'
        }

    def convert_id_to_name(self, id) -> str:
        if id not in self.__id_to_name:
            return 'Unknown'

        return self.__id_to_name[id]