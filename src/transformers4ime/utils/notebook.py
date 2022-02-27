from base64 import b64encode
from io import BytesIO

from PIL import Image as pil_image
from more_itertools import numeric_range


def embedded_image(url):
    pil_im = pil_image.open(url)
    b = BytesIO()
    pil_im.save(b, format='png')
    return "data:image/png;base64,{0}".format(b64encode(b.getvalue()).decode('utf-8'))


def visualize_attrs(tokens, attrs):
    html_text = ""
    for i, tok in enumerate(tokens):
        r, g, b = get_color(attrs[i])
        html_text += " <strong><span style='size:16;color:rgb(%d,%d,%d)'>%s</span></strong>" % (
            r, g, b, tok)
    return html_text


def get_latex(tokens, attrs):
    ans = ""
    for i, tok in enumerate(tokens):
        [r, g, b] = [w / 256.0 for w in get_color(attrs[i])]
        ans += " {\color[rgb]{%f,%f,%f}%s}" % (r, g, b, tok)
    return ans


def normalize_attrs(attrs):
    """ normalize attributions to between -1 and 1 """
    bound = max(abs(attrs.max()), abs(attrs.min()))
    return attrs / bound


def get_color(attr):
    """ attr is assumed to be between -1 and 1 """
    if attr > 0:
        return int(128 * attr) + 127, 128 - int(64 * attr), 128 - int(64 * attr)
    return 128 + int(64 * attr), 128 + int(64 * attr), int(-128 * attr) + 127


def eval_fix(predictions, targets):
    gd_acc = []
    bl_acc = []
    max_acc = 0
    prediction = None
    threshold_list = []

    for t in numeric_range(0.5, 1, 0.05):
        bl = [idx if prob > t else 1 for (prob, idx) in predictions]
        acc = sum(x == y for x, y in zip(bl, targets)) / len(targets)
        bl_acc.append(acc)
        gd_acc.append(sum(x == 1 for x in targets) / len(targets))
        threshold_list.append(t)

        if acc > max_acc:
            max_acc = acc
            prediction = bl
    return max_acc, prediction
