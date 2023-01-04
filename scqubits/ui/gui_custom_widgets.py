# gui_custom_widgets.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################
from typing import Union, List

import ipyvuetify
from IPython.core.display_functions import display


from scqubits.utils import misc as utils


class ValidatedTextFieldABC(ipyvuetify.TextField):
    is_valid: callable = None
    valid_type = None
    _current_value = None

    def __init__(self, **kwargs):
        self.continuous_update_in_progress = False
        self.name = kwargs.pop("name", None)

        if "v_model" not in kwargs and "items" in kwargs:
            kwargs["v_model"] = kwargs["items"][0]
        self._current_value = kwargs["v_model"]

        if "filled" not in kwargs:
            kwargs["filled"] = True

        super().__init__(**kwargs)
        self.observe(self.valid_entry, names="v_model")

    def valid_entry(self, *args, **kwargs):
        if not self.is_valid(self.v_model):
            self.error = True
            self.rules = ["invalid"]
            return False

        self.rules = [True]
        self.error = False
        return True

    @property
    def num_value(self):
        if not self.error:
            self._current_value = self.valid_type(self.v_model)
        return self._current_value


class IntTextField(ValidatedTextFieldABC):
    is_valid = staticmethod(utils.is_string_int)
    valid_type = int
    step = 1


class FloatTextField(ValidatedTextFieldABC):
    is_valid = staticmethod(utils.is_string_float)
    valid_type = float


class NumberEntryWidget:
    def __init__(
        self,
        num_type,
        label,
        min=None,
        step=None,
        max=None,
        v_model=None,
        style_="",
        class_="",
        text_kwargs=None,
        slider_kwargs=None,
    ):
        super().__init__()
        if num_type == float:
            TxtWidget = FloatTextField
            min = min if min is not None else 0.0
            max = max if max is not None else 1.0
            step = step or 0.1
            v_model = v_model if v_model is not None else min
        elif num_type == int:
            TxtWidget = IntTextField
            min = min if min is not None else 0
            max = max if max is not None else 100
            v_model = v_model if v_model is not None else min
        else:
            raise Exception("Unsupported type: ", num_type)

        self.class_ = class_
        self.style_ = style_

        text_kwargs = text_kwargs or {}
        slider_kwargs = slider_kwargs or {}

        self.textfield = TxtWidget(label=label, v_model=v_model, **text_kwargs)

        if "style_" not in slider_kwargs:
            slider_kwargs["style_"] = "max-width: 240px; width: 220px;"
        if "class_" not in slider_kwargs:
            slider_kwargs["class_"] = "pt-3"
        self.slider = ipyvuetify.Slider(
            min=min, max=max, step=step, v_model=v_model, **slider_kwargs
        )
        self.textfield.continuous_update_in_progress = False

        self.slider.on_event("start", self.slider_in_progress_toggle)
        self.slider.on_event("end", self.slider_in_progress_toggle)
        self.slider.observe(self.update_textfield, names="v_model")
        self.textfield.observe(self.update_slider, names="v_model")

    def _ipython_display_(self):
        display(self.widget())

    def widget(self):
        return ipyvuetify.Container(
            class_=self.class_ or "d-flex flex-row pr-4",
            style_=self.style_,
            children=[self.textfield, self.slider],
        )

    def __getattr__(self, item):
        if hasattr(self.textfield, item):
            return getattr(self.textfield, item)
        if hasattr(self.slider, item):
            return getattr(self.slider, item)
        raise Exception(f"NumberEntryWidget.{item} does not exist.")

    def update_textfield(self, *args):
        if self.textfield.continuous_update_in_progress:
            self.textfield.v_model = self.slider.v_model

    def update_slider(self, *args):
        if self.textfield.error:
            return
        if self.textfield.valid_type(self.textfield.v_model) > self.slider.max:
            self.slider.color = "red"
            self.slider.v_model = self.slider.max
        elif self.textfield.valid_type(self.textfield.v_model) < self.slider.min:
            self.slider.color = "red"
            self.slider.v_model = self.slider.min
        else:
            self.slider.color = ""
            self.slider.v_model = self.textfield.valid_type(self.textfield.v_model)

    def slider_in_progress_toggle(self, *args):
        self.textfield.continuous_update_in_progress = (
            not self.textfield.continuous_update_in_progress
        )
        if not self.textfield.continuous_update_in_progress:
            self.textfield.v_model = str(
                self.slider.v_model
            )  # This is a hack... need to trigger final "change" event

    @property
    def disabled(self):
        return self.textfield.disabled

    @disabled.setter
    def disabled(self, value):
        self.textfield.disabled = value
        self.slider.disabled = value

    @property
    def v_model(self):
        return self.textfield.v_model

    @v_model.setter
    def v_model(self, value):
        self.textfield.v_model = value

    @property
    def min(self):
        return self.slider.min

    @min.setter
    def min(self, value):
        self.slider.min = value

    @property
    def max(self):
        return self.slider.max

    @max.setter
    def max(self, value):
        self.slider.max = value

    def observe(self, *args, **kwargs):
        self.textfield.observe(*args, **kwargs)

    def unobserve(self, *args, **kwargs):
        self.textfield.unobserve(*args, **kwargs)


class InitSelect(ipyvuetify.Select):
    def __init__(self, **kwargs):
        if "v_model" not in kwargs and "items" in kwargs:
            kwargs["v_model"] = kwargs["items"][0]
        super().__init__(**kwargs)


class vBtn(ipyvuetify.Btn):
    def __init__(self, **kwargs):
        onclick = kwargs.pop("onclick", None)
        super().__init__(**kwargs)

        if onclick:
            self.on_event("click", onclick)


class DiscreteSetSlider(ipyvuetify.Slider):
    def __init__(self, param_vals, **kwargs):
        self.val_count = len(param_vals)
        self.param_vals = param_vals

        min = 0
        max = self.val_count - 1

        super().__init__(min=min, max=max, step=1, v_model=min, **kwargs)

    def current_value(self):
        return self.param_vals[int(self.v_model)]


class IconButton(vBtn):
    def __init__(self, icon_name, **kwargs):
        super().__init__(
            **kwargs,
            min_width=40,
            width=40,
            height=40,
            elevation="0",
            children=[ipyvuetify.Icon(children=[icon_name])]
        )


class NavbarElement(ipyvuetify.ExpansionPanels):
    def __init__(
        self,
        header,
        content: Union[None, ipyvuetify.ExpansionPanelContent] = None,
        children: Union[None, List[ipyvuetify.VuetifyWidget]] = None,
        **kwargs
    ):
        assert (content and not children) or (children and not content)

        content = (
            content
            if isinstance(content, ipyvuetify.ExpansionPanelContent)
            else ipyvuetify.ExpansionPanelContent(class_="text-no-wrap", style_="transform: scale(0.9)", children=children)
        )

        super().__init__(
            **kwargs,
            **dict(
                transition=False,
                flat=True,
                v_model=None,
                children=[
                    ipyvuetify.ExpansionPanel(
                        accordion=True,
                        children=[
                            ipyvuetify.ExpansionPanelHeader(
                                disable_icon_rotate=True,
                                style_="font-size: 16px; font-weight: 500",
                                class_="text-no-wrap",
                                children=[header],
                            ),
                            content,
                        ],
                    )
                ],
            )
        )
