


def tokens_to_caption(tokens, scale=1) -> str:
    """
    Space tokens based on time gaps without numbering them.
    scale: determines how many spaces to insert per second of gap
    """
    caption = ""
    for i, (t, tok) in enumerate(tokens):
        if i == 0:
            caption += tok
        else:
            prev_t = tokens[i - 1][0]
            gap = t - prev_t
            spaces = ' ' * int(gap * scale)
            caption += spaces + tok
    return caption


