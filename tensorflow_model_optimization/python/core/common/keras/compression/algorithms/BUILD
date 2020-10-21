package(default_visibility = [
    "//tensorflow_model_optimization/python/core/common/keras/compression:__subpackages__",
    "//third_party/tensorflow_model_optimization/python/examples/compression:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "same_training_and_inference",
    srcs = ["same_training_and_inference.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_test(
    name = "same_training_and_inference_test",
    srcs = ["same_training_and_inference_test.py"],
    python_version = "PY3",
    deps = [
        ":same_training_and_inference",
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "different_training_and_inference",
    srcs = ["different_training_and_inference.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_test(
    name = "different_training_and_inference_test",
    srcs = ["different_training_and_inference_test.py"],
    python_version = "PY3",
    deps = [
        ":different_training_and_inference",
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "weight_clustering",
    srcs = ["weight_clustering.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_centroids",
        "//tensorflow_model_optimization/python/core/clustering/keras:clustering_registry",
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_test(
    name = "weight_clustering_test",
    timeout = "eternal",
    srcs = ["weight_clustering_test.py"],
    python_version = "PY3",
    deps = [
        ":weight_clustering",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster_config",
    ],
)
