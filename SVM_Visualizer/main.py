from decision_boundary_visualization import (
    SolutionVisualization,
    polar_kernel_matrix,
    polar_kernel,
    polar_embedding,
)
from support_vector_visualization import (
    SupportVectorVisualization,
    seperate_support_vectors,
)
from utils import load_csv_data, Visualizer, PLOT_SIZE_MAGNIFIER
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


def visualize_embedding(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Creates a plot of the 2d data points, X, embedded in a 3d feature space.
    Also plots the hyperplane in feature space learned by a linear SVM trained on
    the embedded data

    Parameters
    ----------
    X: np.ndarray
        A n_training_samples x n_features numpy matrix of training samples
    Y: np.ndarray
        A numpy vector where each element corresponds to a label of X

    Returns
    -------
    None (Should display 2 plots)
    """
    visualizer = Visualizer()
    gridpoints_x1, gridpoints_x2 = visualizer.get_gridpoints()

    SVM = svm.SVC(kernel="linear") # Creates an SVM object with a linear kernel

    visualize_decision_boundary_embedded_space(X, Y, gridpoints_x1, gridpoints_x2, SVM)
    visualize_decision_boundary_original_space(
        X, Y, gridpoints_x1, gridpoints_x2, visualizer, SVM
    )


def visualize_decision_boundary_embedded_space(
    X: np.ndarray,
    Y: np.ndarray,
    gridpoints_x1: np.ndarray,
    gridpoints_x2: np.ndarray,
    SVM: svm.SVC,
) -> None:
    """
    Creates a 3d plot of the decision boundary in the embedding space
    along with the embedded data points.

    Parameters
    ----------
    X: np.ndarray
        A n_training_samples x n_features numpy matrix of training samples.
    Y: np.ndarray
        A numpy vector where each element corresponds to a label of X.
    gridpoints_x1: np.ndarray
        A n_axis_len x n_axis_len (see utils.py for defition) numpy matrix of values representing x_1 values.
    gridpoints_x2: np.ndarray
        A n_axis_len x n_axis_len numpy matrix of values representing x_2 values.
    SVM: svm
        An sklearn SVM object.

    Returns
    -------
    None (Should display 1 plot)
    """
    decision_boundary_visualization = SolutionVisualization(
        SVM, polar_kernel, polar_embedding
    )

    plot_params_3d = {
        "title": "Higher Dimensional Feature Space",
        "xlabel": r"$x_1$",
        "ylabel": r"$x_2$",
        "zlabel": r"$x_1^2 + x_2^2$",
    }

    # ------ DO NOT MODIFY ANYTHING ABOVE THIS LINE ------

    # Embed the data into the higher dimensional feature space
    embedded_input_data = decision_boundary_visualization.embedding_function(X)
    # Train the svm ON THE EMBEDDED POINTS
    decision_boundary_visualization.svm.fit(embedded_input_data,Y)
    # Get the new feature values
    new_feature_values = decision_boundary_visualization.generate_new_feature_values(gridpoints_x1, gridpoints_x2)

    # This will plot all the (x_1, x_2, x1^2+x2^2) points we've found
    # It will also plot the embedded feature data
    decision_boundary_visualization.visualize_3d(
        gridpoints_x1,
        gridpoints_x2,
        new_feature_values,
        embedded_input_data,
        Y,
        plot_params_3d,
    )


def visualize_decision_boundary_original_space(
    X: np.ndarray,
    Y: np.ndarray,
    gridpoints_x1: np.ndarray,
    gridpoints_x2: np.ndarray,
    visualizer: Visualizer,
    SVM: svm,
) -> None:
    """
    Creates a 2d plot of the decision boundary along with the data points
    in their original space. The decision boundary is the contrast between
    the two different shades of color.

    Parameters
    ----------
    X: np.ndarray
        A n_training_samples x n_features numpy matrix of training samples
    Y: np.ndarray
        A numpy vector where each element corresponds to a label of X
    gridpoints_x1: np.ndarray
        A n_axis_len x n_axis_len (see utils.py for defition) numpy matrix of values representing x_1 values.
    gridpoints_x2: np.ndarray
        A n_axis_len x n_axis_len numpy matrix of values representing x_2 values.
    visualizer: Visualizer
        A Visualizer object.
    SVM: svm
        An sklearn SVM object.

    Returns
    -------
    None (Should display 1 plot)
    """
    plot_params_2d = {
        "title": "Original Feature Space",
        "xlabel": r"$x_1$",
        "ylabel": r"$x_2$",
    }
    gridpoints = np.c_[gridpoints_x1.ravel(), gridpoints_x2.ravel()]
    predictions = SVM.predict(polar_embedding(gridpoints))

    visualizer.visualize_2d(
        gridpoints_x1,
        gridpoints_x2,
        np.expand_dims(predictions, axis=0),
        X,
        Y,
        plot_params_2d,
    )


def visualize_H(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Creates a 3d plot of H(x), one using the dual coefficients learned by
    a linear svm trained on manually embedded data, one using the dual coefficients
    learned by a svm trained using an equivilent kernel.

    Parameters
    ----------
    X: np.ndarray
        A n_training_samples x n_features numpy matrix of training samples
    Y: np.ndarray
        A numpy vector where each element corresponds to a label of X

    Returns
    -------
    None (Should display 2 plots)
    """
    SVM_linear_kernel = svm.SVC(kernel="linear")
    SVM_polar_kernel = svm.SVC(kernel=polar_kernel_matrix)

    linear_H_visualization = SolutionVisualization(
        SVM_linear_kernel, polar_kernel, polar_embedding
    )
    polar_H_visualization = SolutionVisualization(
        SVM_polar_kernel, polar_kernel, polar_embedding
    )

    gridpoints_x1, gridpoints_x2 = linear_H_visualization.get_gridpoints()

    linear_plot_params = {
        "title": "Linear Kernel Trained In Higher Dim Feature Space",
        "xlabel": r"$x_1$",
        "ylabel": r"$x_2$",
        "zlabel": r"$H(x)$",
    }

    polar_plot_params = {param: val for param, val in linear_plot_params.items()}
    polar_plot_params["title"] = "Polar Kernel"

    # ------ DO NOT MODIFY ANYTHING ABOVE THIS LINE ------

    # Embed the data into the higher dimensional feature space
    embedded_input_data = polar_embedding(X)
    # Train the linear kernel svm ON THE EMBEDDED POINTS
    SVM_linear_kernel.fit(embedded_input_data, Y)
    # Train the polar kernel svm on the original data
    SVM_polar_kernel.fit(X, Y)

    linear_H_visualization.visualize_H(
        X, gridpoints_x1, gridpoints_x2, plot_params=linear_plot_params
    )

    polar_H_visualization.visualize_H(
        X, gridpoints_x1, gridpoints_x2, plot_params=polar_plot_params
    )


def visualize_slack_penalty(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Creates multiple 2d contour plots of the decision boundary created by fitting an SVM with an RBF kernel.
    Each contour plot is created using an SVM with a different C value.

    Parameters
    ----------
    X: np.ndarray
        A n_training_samples x n_features numpy matrix of training samples
    Y: np.ndarray
        A numpy vector where each element corresponds to a label of X

    Returns
    -------
    None (Should display 6 plots)
    """
    visualizer = Visualizer()
    solution_points = []

    gridpoints_x1, gridpoints_x2 = visualizer.get_gridpoints()
    gridpoints = np.c_[gridpoints_x1.ravel(), gridpoints_x2.ravel()]

    multi_title = []

    # Define a list of however many C values to train an SVM with.
    C_vals = [0.01, 0.1, 1, 10, 100]

    for c in C_vals:
        multi_title.append(f"c = {c}")

        # Create and fit an SVM object with the built in rbf kernel from sklearn
        rbf_svm = svm.SVC(kernel='rbf', C=c).fit(X,Y)

        # Get the predicted label of each gridpoint
        predictions = rbf_svm.predict(gridpoints)
        solution_points.append(predictions)

    slack_plot_params = {
        "xlabel": r"$x_1$",
        "ylabel": r"$x_2$",
        "multi_title": multi_title,
    }

    visualizer.visualize_2d(
        gridpoints_x1, gridpoints_x2, np.array(solution_points), X, Y, slack_plot_params
    )


def visualize_support_vectors(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Creates several plots of a decision boundary from a linear svm fitted on linearly separable 2d data.
    Also plots the margins, the data, and outlines the support vectors.

    Parameters
    ----------
    X: np.ndarray
        A n_training_samples x n_features numpy matrix of training samples
    Y: np.ndarray
        A numpy vector where each element corresponds to a label of X

    Returns
    -------
    None (Should display multiple plots)
    """
    # The number of times we train the SVM, get the support vectors, then
    # remove them from the training set
    N_REMOVALS = 11

    # The number of iterations inbetween each time the solution is visualized
    # This is basically the frequency of the plots, we plot "once every" n
    # iterations
    ONCE_EVERY = 5

    SVM = svm.SVC(C=1e6, kernel="linear")

    n_plots = int(np.ceil(N_REMOVALS / ONCE_EVERY))
    _, axes = plt.subplots(
        2, n_plots, figsize=(PLOT_SIZE_MAGNIFIER * n_plots, PLOT_SIZE_MAGNIFIER * 2)
    )

    for i in range(N_REMOVALS):
        plot_idx = i // ONCE_EVERY
        should_plot = (i % ONCE_EVERY) == 0

        support_vector_visualization = SupportVectorVisualization(SVM)
        x_axis_points = np.linspace(0, 5)

        SVM.fit(X,Y)

        if should_plot:
            support_vector_visualization.visualize_solution(
                X, Y, x_axis_points, axes[0, plot_idx]
            )

        # Seperate the support vectors and retrain the SVM on just the support vectors
        support_vectors, non_support_vectors = seperate_support_vectors(X, SVM.support_)
        support_vector_labels, non_support_vector_labels = seperate_support_vectors(Y, SVM.support_)

        SVM.fit(support_vectors, support_vector_labels)

        if should_plot:
            support_vector_visualization.visualize_solution(
                support_vectors, support_vector_labels, x_axis_points, axes[1, plot_idx]
            )

        # Remove the support vectors from X and Y
        X = non_support_vectors
        Y = non_support_vector_labels

    plt.show()


def main():
    # ********** PART 1 **********
    circular_data = load_csv_data("data/fake_circular.csv")
    visualize_embedding(circular_data[:, 0:2], circular_data[:, 2])

    # # ********** PART 2 **********
    visualize_H(circular_data[:, 0:2], circular_data[:, 2])

    # ********** PART 3 **********
    nonlinear_data = load_csv_data("data/fake_data.csv")
    visualize_slack_penalty(nonlinear_data[:, 0:2], nonlinear_data[:, 2])

    # ********** PART 4 **********
    linear_data = load_csv_data("data/linearly separable.csv")
    visualize_support_vectors(linear_data[:, 0:2], linear_data[:, 2])


if __name__ == "__main__":
    main()