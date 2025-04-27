from machine_learning import MachineLearning
from data_processing import DataProcessing
from base_main import Main


class ProjectMain(Main):
    def __init__(self):
        super().__init__("projetos.csv")

    def process(self):
        ml = MachineLearning
        dp = DataProcessing

        data = self.data

        data["finalizado"] = dp.map_column(data["nao_finalizado"], {1: 0, 0: 1})

        data = dp.get_query(data, "horas_esperadas > 0")
        dp.get_scatterplot(data, "horas_esperadas", "preco", hue="finalizado")
        dp.show_scatterplot()

        feature = data[["horas_esperadas", "preco"]].values
        target = data["finalizado"].values
        print(type(feature))

        train_x, test_x, train_y, test_y = ml.get_train_test_split(feature, target, test_size=0.25, stratify=target)

        scaler = ml.get_standard_scaler()
        ml.standard_scaler_fit(scaler, train_x)

        scaled_train_x = scaler.transform(train_x)
        scaled_test_x = scaler.transform(test_x)

        linear_svc = ml.get_svc(gamma='auto')
        fitted_svc = ml.fit_svc(linear_svc, scaled_train_x, train_y)

        prediction = ml.predict_linear_svc(fitted_svc, scaled_test_x)
        score = ml.get_accuracy_score(test_y, prediction)
        print(score)


if __name__ == "__main__":
    main = ProjectMain()
    main.process()
