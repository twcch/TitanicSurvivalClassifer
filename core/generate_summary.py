def generate_content(config: dict) -> None:
    lines = []

    lines.append("# Model Summary \n")

    model_info = config.get("model", {})
    lines.append("## Model\n")
    lines.append(f"-    Name: {model_info.get('name')}, version: {model_info.get('version')}")
    lines.append(f"-    Description: {model_info.get('description')}")

    lines.append("\n### Parameters\n")
    for k, v in model_info.get("params", {}).items():
        lines.append(f"-    {k}: {v}")

    lines.append("\n## Features\n")
    features = config.get("features", {})
    lines.append(f"-    Categorical: {features.get('categorical')}")
    lines.append(f"-    Numerical: {features.get('numerical')}")

    lines.append("\n## Preprocessing\n")
    pre = config.get("preprocessing", {})
    lines.append(f"-    Fillna: {pre.get('fillna')}")
    lines.append(f"-    Created Features: {list(pre.get('create_features', {}).keys())}")
    lines.append(f"-    Dropped Features: {pre.get('drop_features')}")

    lines.append("\n## Feature Engineering\n")
    fe = config.get("feature_engineering", {})
    enc = fe.get("encodings", {}).get("onehot", {})
    lines.append(f"-    OneHot Features: {enc.get('features')}")
    lines.append(f"-    OneHot Params: {enc.get('params')}")
    lines.append(f"-    Scaling: {fe.get('scaling')}")

    lines.append("\n## Training\n")
    training = config.get("training", {})
    lines.append(f"-    Train Size: {training.get('train_size')}")
    lines.append(f"-    Test Size: {training.get('test_size')}")
    lines.append(f"-    Metrics: {training.get('evaluation', {}).get('metrics')}")

    lines.append("\n## Output\n")
    output = config.get("output", {})
    lines.append(f"-    Model Path: `{output.get('model_path')}`")
    lines.append(f"-    Submission File: `{output.get('submission_file')}`")

    return "\n".join(lines)


def get_content(config):
    content = generate_content(config)

    return content
