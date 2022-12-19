import ipyvuetify

from scqubits.utils import misc as utils


class ValidatedTextFieldABC(ipyvuetify.TextField):
    is_valid: callable = None
    valid_type = None

    def __init__(self, onchange, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "name" in kwargs.keys()

        self.onchange_callback = onchange

        self.param_name = kwargs["name"]

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
