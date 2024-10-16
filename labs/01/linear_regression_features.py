#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import PolynomialFeatures

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=10, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--range", default=6, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> list[float]:
    # Create artificial noisy data.
    xs = np.linspace(0, 7, num=args.data_size).reshape(-1,1)
    ys = np.sin(xs).ravel() + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)

    # Here i had to change ys and xs, as it was here originally, it did not work properly and the output was incorrect
    # I dont know if we were supposed to do that but yeah.abs
    # ORIGINAL
    # xs = np.linspace(0, 7, num=args.data_size)
    # ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)
    # ....
    # X = np.vander(xs, order, increasing=True)  

    rmses = []
    for order in range(1, args.range + 1):
        # TODO: Create features `(x^1, x^2, ..., x^order)`, preferably in this ordering.
        # Note that you can just append `x^order` to the features from the previous iteration.
        poly = PolynomialFeatures(degree=order)
        X = poly.fit_transform(xs)
        
        # TODO: Split the data into a train set and a test set.
        # Use `sklearn.model_selection.train_test_split` method call, passing
        # arguments `test_size=args.test_size, random_state=args.seed`.
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, ys, test_size=args.test_size, random_state=args.seed
        )

        # TODO: Fit a linear regression model using `sklearn.linear_model.LinearRegression`;
        # consult the documentation and see especially the `fit` method.
        model = sklearn.linear_model.LinearRegression()
        model.fit(X_train, y_train)

        # TODO: Predict targets on the test set using the `predict` method of the trained model.
        y_pred = model.predict(X_test)

        # TODO: Compute root mean square error on the test set predictions.
        # You can either do it manually, or you can look at the metrics offered
        # by the `sklearn.metrics` module.
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))
        rmses.append(rmse)

        if args.plot:
            # The plotting code assumes the train/test data/targets are in numpy arrays
            # `train_data`, `train_target`, `test_data`, `test_target`.
            train_data, train_target = X_train, y_train
            test_data, test_target = X_test, y_test
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.plot(train_data[:, 0], train_target, "go")
            plt.plot(test_data[:, 0], test_target, "ro")
            plt.plot(np.linspace(xs[0], xs[-1], num=100),
                     model.predict(np.power.outer(np.linspace(xs[0], xs[-1], num=100), np.arange(1, order + 1))), "b")
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return rmses


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    rmses = main(main_args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))
