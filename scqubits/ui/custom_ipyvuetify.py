import ipyvuetify
import ipywidgets

from scqubits.utils import misc as utils


class ValidatedTextFieldABC(ipyvuetify.TextField):
    is_valid: callable = None
    valid_type = None

    def __init__(self, *args, onchange=None, **kwargs):
        super().__init__(*args, **kwargs)
        if "label" in kwargs.keys() and not "name" in kwargs.keys():
            kwargs["name"] = kwargs["label"]

        self.onchange_callback = onchange

        self.param_name = kwargs["name"]

        if onchange:
            self.on_event("change", self.safe_callback)
        self.on_event("input", self.valid_entry)

    def valid_entry(self, *args, **kwargs):
        if not self.is_valid(self.v_model):
            self.rules = ["Error: invalid entry"]
            return False

        self.rules = [True]
        return True

    def safe_callback(self, *args, **kwargs):
        if self.valid_entry():
            self.onchange_callback(**{self.param_name: self.v_model})
        else:
            pass


class IntTextField(ValidatedTextFieldABC):
    is_valid = staticmethod(utils.is_string_int)
    valid_type = int


class FloatTextField(ValidatedTextFieldABC):
    is_valid = staticmethod(utils.is_string_float)
    valid_type = float


def make_slider_textfield(text_kwargs=None, **slider_kwargs):

    if "style_" not in slider_kwargs:
        slider_kwargs["style_"] = "width: 200px;"

    slider = ipyvuetify.Slider(**slider_kwargs)

    text_kwargs = text_kwargs or {}

    if "style_" not in text_kwargs:
        text_kwargs["style_"] = ""
    text_kwargs["style_"] += "max-width: 50px; width: 50px;"

    text_field = IntTextField(
        v_model=slider_kwargs["v_model"], name=slider_kwargs["label"],
        **text_kwargs
    )

    ipywidgets.jslink((slider, "v_model"), (text_field, "v_model"))
    return slider, text_field


def make_slider_floattextfield(text_kwargs=None, min=0.0, max=1.0,
                               steps=100, **slider_kwargs):

    if "style_" not in slider_kwargs:
        slider_kwargs["style_"] = "width: 200px;"

    slider = ipyvuetify.Slider(min=0, max=steps, **slider_kwargs)

    text_kwargs = text_kwargs or {}

    if "style_" not in text_kwargs:
        text_kwargs["style_"] = ""
    text_kwargs["style_"] += "max-width: 50px; width: 50px;"

    text_field = FloatTextField(
        v_model=slider_kwargs["v_model"], name=slider_kwargs["label"],
        **text_kwargs
    )

    ipywidgets.jslink((slider, "v_model"), (text_field, "v_model"))
    return slider, text_field