from larq_swarm import register_preprocess


@register_preprocess("mnist")
def default(image):
    return image
