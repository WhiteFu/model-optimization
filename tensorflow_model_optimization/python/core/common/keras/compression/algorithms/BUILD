package(default_visibility = ["//visibility:private"])

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
    timeout = "long",
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
    timeout = "long",
    srcs = ["different_training_and_inference_test.py"],
    python_version = "PY3",
    deps = [
        ":different_training_and_inference",
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "bias_only",
    srcs = ["bias_only.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/common/keras/compression:algorithm",
    ],
)

py_test(
    name = "bias_only_test",
    timeout = "long",
    srcs = ["bias_only_test.py"],
    python_version = "PY3",
    deps = [
        ":bias_only",
        # numpy dep1,
        # tensorflow dep1,
    ],
)
