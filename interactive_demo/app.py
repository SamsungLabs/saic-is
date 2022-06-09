import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import cv2
import numpy as np
from PIL import Image

from interactive_demo.canvas import CanvasImage, IStatus
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import (
    BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton,
    FocusButton, FocusLabelFrame
)
from isegm.data.interaction_type import IType
from isegm.utils.misc import limit_longest_size


class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, model, cfg):
        super().__init__(master)
        self.master = master
        master.title("Interactive Image Segmentation")
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.limit_longest_size = args.limit_longest_size
        self.fixed_size = (args.fixed_h, args.fixed_w) if args.fixed_h is not None else None

        self.max_size = args.max_size
        self.orig_shape = None

        self.model_input_type = [IType.point]
        if hasattr(model, 'input_type') and model.input_type:
            if isinstance(model.input_type, str):
                model.input_type = IType[model.input_type]
            if isinstance(model.input_type, list):
                self.model_input_type = [IType[itype.name] for itype in model.input_type]
            else:
                self.model_input_type = [IType[model.input_type.name]]
        self.itypes = [itype.name for itype in self.model_input_type]

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={},
                                                update_image_callback=self._update_image,
                                                model_input_type=self.model_input_type,
                                                contour_filled=cfg.MODEL.CONTOUR_FILLED)

        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_buttons()

        master.bind('<space>', lambda event: self.controller.finish_object())

        self.state['zoomin_params']['skip_interactions'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._reset_predictor)
        self._change_itype()
        self.interm_state = None

    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_interactions': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=0),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },

            'predictor_params': {
                'net_interactions_limit': tk.IntVar(value=8)
            },
            'itype': tk.StringVar(value=self.itypes[0]),
            'prob_thresh': tk.DoubleVar(value=0.5),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=3),
            'line_width': tk.IntVar(value=10),
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT)
        self.save_mask_btn = FocusButton(self.menubar, text='Save mask', command=self._save_mask_callback)
        self.save_mask_btn.pack(side=tk.LEFT)
        self.save_mask_btn.configure(state=tk.DISABLED)

        self.load_mask_btn = FocusButton(self.menubar, text='Load mask', command=self._load_mask_callback)
        self.load_mask_btn.pack(side=tk.LEFT)
        self.load_mask_btn.configure(state=tk.DISABLED)

        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        self.input_type_label = tk.Label(self.control_frame, text=f"Input type: {self.itypes}")
        self.input_type_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        self.interaction_type_frame = FocusLabelFrame(master, text="Interaction type choice")
        self.interaction_type_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        menu = tk.OptionMenu(
            self.interaction_type_frame,
            self.state['itype'], *self.itypes,
            command=self._change_itype
        )
        menu.config(width=11)
        menu.grid(rowspan=2, column=0, padx=10)

        self.interaction_options_frame = FocusLabelFrame(master, text="Interaction management")
        self.interaction_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.finish_object_button = FocusButton(
            self.interaction_options_frame,
            text='Finish\nobject', bg='#b6d7a8', fg='black', width=10, height=2,
            state=tk.DISABLED, command=self.controller.finish_object
        )
        self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.undo_last_button = FocusButton(
            self.interaction_options_frame,
            text='Undo\nlast', bg='#ffe599', fg='black', width=10, height=2,
            state=tk.DISABLED, command=self.controller.undo_last
        )
        self.undo_last_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_button = FocusButton(
            self.interaction_options_frame,
            text='Reset', bg='#ea9999', fg='black', width=10, height=2,
            state=tk.DISABLED, command=self._reset_last_object
        )
        self.reset_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.zoomin_options_frame = FocusLabelFrame(master, text="ZoomIn options")
        self.zoomin_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusCheckButton(self.zoomin_options_frame, text='Use ZoomIn', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['use_zoom_in']).grid(row=0, column=0, padx=10)
        FocusCheckButton(self.zoomin_options_frame, text='Fixed crop', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['fixed_crop']).grid(row=1, column=0, padx=10)
        tk.Label(self.zoomin_options_frame, text="Skip interactions").grid(row=0, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Target size").grid(row=1, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Expand ratio").grid(row=2, column=1, pady=1, sticky='e')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['skip_interactions'],
                              min_value=-1, max_value=None, vartype=int,
                              name='zoom_in_skip_interactions').grid(row=0, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['target_size'],
                              min_value=0, max_value=self.limit_longest_size, vartype=int,
                              name='zoom_in_target_size').grid(row=1, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['expansion_ratio'],
                              min_value=1.0, max_value=2.0, vartype=float,
                              name='zoom_in_expansion_ratio').grid(row=2, column=2, padx=10, pady=1, sticky='w')
        self.zoomin_options_frame.columnconfigure((0, 1, 2), weight=1)

        self.prob_thresh_frame = FocusLabelFrame(master, text="Predictions threshold")
        self.prob_thresh_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.prob_thresh_frame, from_=0.0, to=1.0, command=self._update_prob_thresh,
                             variable=self.state['prob_thresh']).pack(padx=10)

        self.alpha_blend_frame = FocusLabelFrame(master, text="Alpha blending coefficient")
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.alpha_blend_frame, from_=0.0, to=1.0, command=self._update_blend_alpha,
                             variable=self.state['alpha_blend']).pack(padx=10, anchor=tk.CENTER)

        self.click_radius_frame = FocusLabelFrame(master, text="Visualisation click radius")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=7, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor=tk.CENTER)

        self.line_width_frame = FocusLabelFrame(master, text="Visualisation lines width")
        self.line_width_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(
            self.line_width_frame, from_=1, to=14, resolution=1,
            command=self._update_line_width,
            variable=self.state['line_width']
        ).pack(padx=10, anchor=tk.CENTER)

    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ], title="Choose an image")

            if len(filename) > 0:
                image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                self.orig_shape = image.shape[:2]
                if self.max_size is not None:
                    image, new_shape = limit_longest_size(image, self.max_size)
                self.controller.set_image(image)
                self.save_mask_btn.configure(state=tk.NORMAL)
                self.load_mask_btn.configure(state=tk.NORMAL)

    def _save_mask_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            mask = self.controller.result_mask
            if mask is None:
                return

            filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
                ("PNG image", "*.png"),
                ("BMP image", "*.bmp"),
                ("All files", "*.*"),
            ], title="Save the current mask as...")

            if len(filename) > 0:
                if mask.max() < 256:
                    mask = mask.astype(np.uint8)
                    mask *= 255 // mask.max()
                if self.max_size is not None:
                    mask, new_shape = limit_longest_size(mask, None, self.orig_shape, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(filename, mask)

    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning("Warning", "The current model doesn't support loading external masks. "
                                              "Please use ITER-M models for that purpose.")
            return

        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Binary mask (png, bmp)", "*.png *.bmp"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                mask = cv2.imread(filename)
                if self.max_size is not None:
                    mask, new_shape = limit_longest_size(mask, self.max_size, None, cv2.INTER_NEAREST)
                mask = mask[:, :, 0] > 127
                self.controller.set_mask(mask)
                self._update_image()

    def _about_callback(self):
        self.menubar.focus_set()
        text = ["The MIT License, 2021"]
        messagebox.showinfo("About Demo", '\n'.join(text))

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, *args):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, *args):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return
        self._update_image()

    def _update_line_width(self, *args):
        if self.image_on_canvas is None:
            return
        self._update_image()

    def _change_itype(self, *args):
        self.controller.input_type = IType[self.state['itype'].get()]

    def _reset_predictor(self, *args, **kwargs):
        prob_thresh = self.state['prob_thresh'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get(),
                'skip_interactions': self.state['zoomin_params']['skip_interactions'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_interactions_limit': None,
                'max_size': self.limit_longest_size,
                'fixed_size': self.fixed_size,
            },
        }
        self.controller.reset_predictor(predictor_params)

    def _callback(self, status, **kwargs):
        self.canvas.focus_set()
        current_itype = IType[self.state['itype'].get()]

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            if status == IStatus.begin:
                if current_itype == IType.point:
                    self.controller.add_interaction(**kwargs)
                else:
                    self.controller.begin_interaction(**kwargs)
            elif status == IStatus.motion and current_itype != IType.point:
                self.controller.update_interaction(**kwargs)
            elif status == IStatus.end and current_itype != IType.point:
                self.controller.add_interaction(**kwargs)

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(
            alpha_blend=self.state['alpha_blend'].get(),
            click_radius=self.state['click_radius'].get(),
            line_width=self.state['line_width'].get()
        )
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas, self)
            self.image_on_canvas.register_callback(self._callback)

        self._set_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _set_widgets_state(self):
        after_1st_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
        before_1st_state = tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL

        self.finish_object_button.configure(state=after_1st_state)
        self.undo_last_button.configure(state=after_1st_state)
        self.reset_button.configure(state=after_1st_state)
        self.zoomin_options_frame.set_frame_state(before_1st_state)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked
