from machine_learning import MachineLearning
from data_processing import DataProcessing
from base_main import Main


class ProjectMain(Main):
    def __init__(self):
        super().__init__("tracking.csv")

    def process(self):
        ml = MachineLearning()
        dp = DataProcessing()

        data = dp.get_csv(self.filepath)

        feature = data[["inicial", "palestras", "contato", "patrocinio"]].values
        target = data["comprou"].values

        x_train, x_test, y_train, y_test = ml.get_train_test_split(feature, target, stratify=target)

        linear_svc = ml.get_linear_svc()
        linear_svc_fit = ml.fit_linear_svc(linear_svc, x_train, y_train)
        prediction = ml.predict_linear_svc(linear_svc_fit, x_test)
        print(ml.get_accuracy_score(y_test, prediction))


if __name__ == "__main__":
    main = ProjectMain()
    main.process()
