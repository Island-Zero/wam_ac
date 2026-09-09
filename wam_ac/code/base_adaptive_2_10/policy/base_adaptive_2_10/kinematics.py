from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _tf(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def _rpy(text: str | None) -> np.ndarray:
    return Rotation.from_euler("xyz", np.fromstring(text or "0 0 0", sep=" ")).as_matrix()


@dataclass(frozen=True)
class Joint:
    parent: str
    child: str
    position: np.ndarray
    rotation: np.ndarray
    axis: np.ndarray
    kind: str
    name: str


class URDFKinematics:
    def __init__(self, path: str | Path):
        root = ET.parse(path).getroot()
        self.children: dict[str, list[Joint]] = {}
        joints = []
        for node in root.findall("joint"):
            origin, axis = node.find("origin"), node.find("axis")
            joint = Joint(
                node.find("parent").attrib["link"], node.find("child").attrib["link"],
                np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3),
                _rpy(origin.attrib.get("rpy") if origin is not None else None),
                np.fromstring(axis.attrib.get("xyz", "0 0 1"), sep=" ") if axis is not None else np.array([0., 0., 1.]),
                node.attrib.get("type", "fixed"), node.attrib["name"],
            )
            joints.append(joint)
            self.children.setdefault(joint.parent, []).append(joint)
        child_links = {j.child for j in joints}
        roots = sorted({j.parent for j in joints} - child_links)
        if not roots:
            raise ValueError(f"No URDF root link: {path}")
        self.root = roots[0]

    def transforms(self, q: dict[str, float], root_tf: np.ndarray) -> dict[str, np.ndarray]:
        result, stack = {self.root: root_tf}, [self.root]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                transform = result[parent] @ _tf(joint.rotation, joint.position)
                value = float(q.get(joint.name, 0.0))
                if joint.kind in ("revolute", "continuous"):
                    axis = joint.axis / (np.linalg.norm(joint.axis) + 1e-12)
                    transform = transform @ _tf(Rotation.from_rotvec(axis * value).as_matrix(), np.zeros(3))
                elif joint.kind == "prismatic":
                    transform = transform @ _tf(np.eye(3), joint.axis * value)
                result[joint.child] = transform
                stack.append(joint.child)
        return result


@lru_cache(maxsize=4)
def _load(path: str) -> URDFKinematics:
    return URDFKinematics(path)


def aloha_fk(qpos: np.ndarray, urdf: str | Path) -> dict[str, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None]
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"Expected [T,14] qpos, got {qpos.shape}")
    kin = _load(str(Path(urdf).resolve()))
    # Frozen RoboTwin Aloha-AgileX root pose. SAPIEN quaternion is wxyz.
    root_rotation = Rotation.from_quat([0.0, 0.0, 0.707, 0.707]).as_matrix()
    root_tf = _tf(root_rotation, np.array([0.0, -0.65, 0.0]))
    names = {
        "left": ([f"fl_joint{i}" for i in range(1, 7)], slice(0, 6), "fl_link6"),
        "right": ([f"fr_joint{i}" for i in range(1, 7)], slice(7, 13), "fr_link6"),
    }
    output = {"left": [], "right": []}
    for row in qpos:
        values = {}
        for _, (joint_names, indices, _) in names.items():
            values.update(dict(zip(joint_names, row[indices])))
        links = kin.transforms(values, root_tf)
        for arm, (_, _, link) in names.items():
            if link not in links:
                raise KeyError(f"URDF lacks required wrist link: {link}")
            output[arm].append(links[link])
    return {arm: np.stack(items) for arm, items in output.items()}


def execution_position_mean_m(a: np.ndarray, b: np.ndarray, urdf: str | Path, e: int = 24) -> float:
    af, bf = aloha_fk(a, urdf), aloha_fk(b, urdf)
    distances = [np.linalg.norm(af[arm][:e, :3, 3] - bf[arm][:e, :3, 3], axis=1)
                 for arm in ("left", "right")]
    return float(np.concatenate(distances).mean())
