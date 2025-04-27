from machine_learning import MachineLearning
from data_processing import DataProcessing
from base_main import Main


class ProjectMain(Main):
    def __init__(self):
        super().__init__("precos.csv")

    def process(self):
        ml = MachineLearning
        dp = DataProcessing
        data = self.data

        data = dp.convert_miles_to_km(data, "milhas_por_ano", "km_por_ano")
        dp.get_age(data, "ano_do_modelo", "idade")
        dp.drop_columns(data, "milhas_por_ano")

        feature = data[["preco", "idade", "km_por_ano"]].values
        target = data["vendido"].values

        x_train, x_test, y_train, y_test = ml.get_train_test_split(feature, target, stratify=target)

        scaler = ml.get_standard_scaler()
        ml.standard_scaler_fit(scaler, x_train)

        scaled_x_train = scaler.transform(x_train)
        scaled_x_test = scaler.transform(x_test)

        dummy_classifier = ml.get_dummy(strategy="stratified")
        ml.fit_dummy_classifier(dummy_classifier, scaled_x_train, y_train)
        prediction = ml.predict_dummy_classifier(dummy_classifier, scaled_x_test)
        print(ml.get_accuracy_score(y_test, prediction))

        decision_tree_classifier = ml.get_decision_tree()
        ml.fit_decision_tree_classifier(decision_tree_classifier, scaled_x_train, y_train)
        prediction = ml.predict_decision_tree_classifier(decision_tree_classifier, scaled_x_test)
        print(ml.get_accuracy_score(y_test, prediction))


if __name__ == "__main__":
    main = ProjectMain()
    main.process()
