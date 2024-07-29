from keras.models import model_from_json
from keras import optimizers

class ResumeCNNModel:
    def __init__(self) -> None:
        self.cv_height = 500
        self.models_dir = "app/models/"
        self.classes = {
            "Software_Developer": 0,
            "Front_End_Developer": 1,
            "Network_Administrator": 2,
            "Web_Developer": 3,
            "Project_manager": 4,
            "Database_Administrator": 5,
            "Security_Analyst": 6,
            "Systems_Administrator": 7,
            "Python_Developer": 8,
            "Java_Developer": 9,
        }
        self.labels = [
            "Software_Developer",
            "Front_End_Developer",
            "Network_Administrator",
            "Web_Developer",
            "Project_manager",
            "Database_Administrator",
            "Security_Analyst",
            "Systems_Administrator",
            "Python_Developer",
            "Java_Developer",
        ]

    def build_model(self):
        multi_label_model = dict()
        for label in self.labels:
            json_file = open(self.models_dir + label + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            sgd = optimizers.SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.1, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model.load_weights(self.models_dir + label + '.h5')
            multi_label_model[label] = model
        return multi_label_model
