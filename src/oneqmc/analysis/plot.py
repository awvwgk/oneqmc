import colour
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import Colormap, LinearSegmentedColormap


def set_axes():
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.top"] = False
    rcParams["axes.edgecolor"] = "black"


def set_figure():
    rcParams["figure.dpi"] = 100
    rcParams["figure.figsize"] = (6, 4)


def set_savefig():
    rcParams["savefig.format"] = "pdf"
    rcParams["savefig.transparent"] = True


def set_markers():
    semi_transparent_white = "(1, 1, 1, 0.3)"  # Used only for the edge of markers
    rcParams["lines.markeredgecolor"] = semi_transparent_white
    rcParams["lines.markersize"] = 7


def set_font(fontsize: int = 14):
    rcParams["font.size"] = fontsize
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42


def set_defaults(fontsize: int = 14):
    set_axes()
    set_figure()
    set_markers()
    set_font(fontsize)


def get_cyclic_cmap(
    colours_hex: list[str],
    luminosity_min: float = 0.2,
    luminosity_max: float = 0.9,
    name: str = "custom_cmap",
) -> Colormap:
    assert len(colours_hex) > 1, "Cyclic colormap must have at least 2 colours"
    assert len(colours_hex) % 2 == 0, "Cyclic colormap must have an even number of colours"

    standard_cmap = LinearSegmentedColormap.from_list(
        name, colours_hex[: len(colours_hex) // 2 + 1]
    )
    cmap_rgb = [f[:3] for f in standard_cmap(np.linspace(0, 1, 256))]
    cmap_cam = colour.convert(cmap_rgb, "RGB", "CAM16")
    cmap_cam_pu_up = colour.CAM_Specification_CAM16(
        J=np.linspace(luminosity_min, luminosity_max, cmap_cam.J.shape[0]),
        C=cmap_cam.C,
        h=cmap_cam.h,
    )

    cmap_rgb_pu_up = colour.convert(cmap_cam_pu_up, "CAM16", "RGB")
    cmap_rgb_pu_up = np.concatenate([cmap_rgb_pu_up, np.ones((cmap_rgb_pu_up.shape[0], 1))], axis=1)

    standard_cmap = LinearSegmentedColormap.from_list(
        name, colours_hex[len(colours_hex) // 2 :] + [colours_hex[0]]
    )
    cmap_rgb = [f[:3] for f in standard_cmap(np.linspace(0, 1, 256))]
    cmap_cam = colour.convert(cmap_rgb, "RGB", "CAM16")
    cmap_cam_pu_down = colour.CAM_Specification_CAM16(
        J=np.linspace(luminosity_max, luminosity_min, cmap_cam.J.shape[0]),
        C=cmap_cam.C,
        h=cmap_cam.h,
    )

    cmap_rgb_pu_down = colour.convert(cmap_cam_pu_down, "CAM16", "RGB")
    cmap_rgb_pu_down = np.concatenate(
        [cmap_rgb_pu_down, np.ones((cmap_rgb_pu_down.shape[0], 1))], axis=1
    )

    cmap_rgb_pu_cyclic = np.concatenate([cmap_rgb_pu_up, cmap_rgb_pu_down], axis=0)
    cmap_rgb_pu_cyclic = np.clip(cmap_rgb_pu_cyclic, 0, 1)
    return LinearSegmentedColormap.from_list(name, cmap_rgb_pu_cyclic)
