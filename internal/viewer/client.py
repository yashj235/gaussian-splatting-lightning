import time
import threading
import traceback
import numpy as np
import torch
import viser
import viser.transforms as vtf
from internal.cameras.cameras import Cameras
from internal.utils.graphics_utils import fov2focal


class ClientThread(threading.Thread):
    def __init__(self, viewer, renderer, client: viser.ClientHandle):
        super().__init__()
        self.viewer = viewer
        self.renderer = renderer
        self.client = client

        self.render_trigger = threading.Event()

        self.last_move_time = 0

        self.last_camera = None  # store camera information

        self.state = "low"  # low or high render resolution

        self.stop_client = False  # whether stop this thread

        if viewer.default_camera_position is not None:
            client.camera.position = np.asarray(viewer.default_camera_position)
        if viewer.default_camera_look_at is not None:
            client.camera.look_at = np.asarray(viewer.default_camera_look_at)

        client.camera.up_direction = viewer.up_direction

        @client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            with self.client.atomic():
                self.last_camera = cam
                self.state = "low"  # switch to low resolution mode when a new camera received
                self.render_trigger.set()

    @classmethod
    def get_camera(
            cls,
            camera: viser.CameraHandle,
            image_size: int,
            appearance_id: tuple[int, float] = (0, 0),
            time_value: float = 0.,
            camera_transform=None,
    ):
        # calculate image width and height
        image_height = image_size
        image_width = int(image_height * camera.aspect)
        if image_width > image_size:
            image_width = image_size
            image_height = int(image_width / camera.aspect)

        # get camera pose
        R = vtf.SO3(wxyz=camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(camera.position, dtype=torch.float64)
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos

        if camera_transform is not None:
            c2w = torch.matmul(camera_transform, c2w)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        # construct camera
        fx = torch.tensor([fov2focal(camera.fov, max(image_width, image_height))], dtype=torch.float)
        camera = Cameras(
            R=R.unsqueeze(0),
            T=T.unsqueeze(0),
            fx=fx,
            fy=fx,
            cx=torch.tensor([(image_width // 2)], dtype=torch.int),
            cy=torch.tensor([(image_height // 2)], dtype=torch.int),
            width=torch.tensor([image_width], dtype=torch.int),
            height=torch.tensor([image_height], dtype=torch.int),
            appearance_id=torch.tensor([appearance_id[0]], dtype=torch.int),
            normalized_appearance_id=torch.tensor([appearance_id[1]], dtype=torch.float),
            time=torch.tensor([time_value], dtype=torch.float),
            distortion_params=None,
            camera_type=torch.tensor([0], dtype=torch.int),
        )[0]

        return camera

    def render_and_send(self):
        with self.client.atomic():
            self.last_move_time = time.time()

            max_res, jpeg_quality = self.get_render_options()
            camera = self.get_camera(
                self.client.camera,
                image_size=max_res,
                appearance_id=self.viewer.get_appearance_id_value(),
                time_value=self.viewer.time_slider.value,
                camera_transform=self.viewer.camera_transform,
            ).to_device(self.viewer.device)

        with torch.no_grad():
            try:
                image = self.renderer.get_outputs(camera, scaling_modifier=self.viewer.scaling_modifier.value)
                image = torch.clamp(image, max=1.)
                image = torch.permute(image, (1, 2, 0))
            except:
                traceback.print_exc()
                return
            self.client.set_background_image(
                image.cpu().numpy(),
                format=self.viewer.image_format,
                jpeg_quality=jpeg_quality,
            )

    def run(self):
        while True:
            trigger_wait_return = self.render_trigger.wait(0.2)  # TODO: avoid wasting CPU
            # stop client thread?
            if self.stop_client is True:
                break
            if not trigger_wait_return:
                # skip if camera is none
                if self.last_camera is None:
                    continue

                # if we haven't received a trigger in a while, switch to high resolution
                if self.state == "low":
                    self.state = "high"  # switch to high resolution mode
                    # skip rendering if resolution not higher
                    if self.viewer.max_res_when_moving.value >= self.viewer.max_res_when_static.value and self.viewer.jpeg_quality_when_moving.value >= self.viewer.jpeg_quality_when_static.value:
                        continue
                else:
                    continue  # skip if already in high resolution mode

            self.render_trigger.clear()

            try:
                self.render_and_send()
            except Exception as err:
                print("error occurred when rendering for client")
                traceback.print_exc()
                break

        self._destroy()

    def get_render_options(self):
        if self.state == "low":
            return self.viewer.max_res_when_moving.value, int(self.viewer.jpeg_quality_when_moving.value)
        return self.viewer.max_res_when_static.value, int(self.viewer.jpeg_quality_when_static.value)

    def stop(self):
        self.stop_client = True
        # self.render_trigger.set()  # TODO: potential thread leakage?

    def _destroy(self):
        print("client thread #{} destroyed".format(self.client.client_id))
        self.viewer = None
        self.renderer = None
        self.client = None
        self.last_camera = None
