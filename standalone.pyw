from dataclasses import dataclass
from math import exp, ceil
import os
from shutil import rmtree
import time
import tkinter as tk
from tkinter import ttk

from anastruct import SystemElements, Vertex
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

LARGE_FONT = ("Verdana", 12)
RESOLUTION = '750x700'


# analysis.py

@dataclass
class BeamAnalysis:
    name: str
    cases: dict

    def get_graph_dataframe(self):
        df = pd.DataFrame()
        load_arr = [0, ]
        moment_arr = [0, ]
        branson_arr = [0, ]
        bischoff_arr = [0, ]
        mef_arr = [0, ]
        for c in self.cases.values():
            load_arr.append(c.load)
            moment_arr.append(c.get_max_moment())
            branson_arr.append(c.branson)
            bischoff_arr.append(c.bischoff)
            mef_arr.append(c.get_max_deflection())
        df['load'] = load_arr
        df['moment'] = moment_arr
        df['branson'] = branson_arr
        df['bischoff'] = bischoff_arr
        df['mef'] = mef_arr
        return df

    def get_bending_diagram_dataframe(self, total_length):
        df = pd.DataFrame()
        last_case_key = max([k for k in self.cases.keys()])
        last_case = self.cases[last_case_key]
        bar_length = total_length / len(last_case.bars.keys())
        len_arr = []
        moment_arr = []
        for v in last_case.bars.values():
            len_arr.append((v.nodes[0].id - 1) * bar_length)
            moment_arr.append(v.nodes[0].M)
            if v.id == max([key for key in last_case.bars.keys()]):
                len_arr.append((v.nodes[1].id - 1) * bar_length)
                moment_arr.append(v.nodes[1].M)
        df['length'] = len_arr
        df['moment'] = moment_arr
        return df

    def get_shear_diagram_dataframe(self, total_length):
        df = pd.DataFrame()
        last_case_key = max([k for k in self.cases.keys()])
        last_case = self.cases[last_case_key]
        bar_length = total_length / len(last_case.bars.keys())
        len_arr = []
        shear_arr = []
        for v in last_case.bars.values():
            len_arr.append((v.nodes[0].id - 1) * bar_length)
            shear_arr.append(v.nodes[0].V)
            if v.id == max([key for key in last_case.bars.keys()]):
                len_arr.append((v.nodes[1].id - 1) * bar_length)
                shear_arr.append(v.nodes[1].V)
        df['length'] = len_arr
        df['shear'] = shear_arr
        return df


@dataclass
class LoadCaseObject:
    load: float
    bars: dict
    branson: float
    bischoff: float

    def get_node_deflection(self, node_id):
        return

    def get_node_moment(self):
        pass

    def get_max_deflection(self):
        max_uy = 0
        for b in self.bars.values():
            for n in b.nodes.values():
                if n.uy > max_uy:
                    max_uy = n.uy
        return max_uy

    def get_max_moment(self):
        max_moment = 0
        for b in self.bars.values():
            for n in b.nodes.values():
                if abs(n.M) > max_moment:
                    max_moment = abs(n.M)
        return max_moment


@dataclass
class NodeObject:
    id: int
    V: float
    M: float
    uy: float
    phi: float


@dataclass
class BarObject:
    id: int
    EI: float
    EA: float
    cracked: bool
    nodes: dict

    def get_max_deflection(self):
        return max([n.uy for n in self.nodes.values()])

    def get_max_shear(self):
        return max([n.uy for n in self.nodes.values()])


# materials.py

@dataclass
class Rebar:
    fyk: int
    ys: float
    gamma: int
    Es: int
    As: int
    As_neg: int


@dataclass
class Concrete:
    fck: int
    yc: float
    gamma: int
    CP: int
    v: float = 0.20
    alpha: float = 0.00001

    def fcd(self):
        return self.fckj() / self.yc

    def fctm(self):
        return 0.3 * self.fckj() ** (2 / 3)

    def fctk_inf(self):
        return 0.7 * self.fctm()

    def fctk_sup(self):
        return 1, 3 * self.fctm()

    def eci(self):
        return 5600 * (self.fckj() ** (1 / 2))

    def ecs(self):
        return (0.8 + 0.2 * (self.fckj() / 80)) * self.ecij()

    def gc(self):
        return 0.4 * self.ecs()

    def fckj(self, days=28):
        """
        s = 0,38 para concreto de cimento CPIII e CPIV;
        s = 0,25 para concreto de cimento CPI e CPII
        s = 0,20 para concreto de cimento CPV-ARI; fonte NBR 6118:2014
        :param days: int
        :return: int
        """
        if self.CP < 1 or self.CP > 5:
            return -1
        elif self.CP < 3:
            s = 0.25
        elif self.CP < 5:
            s = 0.38
        else:
            s = 0.20
        betta = exp(s * (1 - (28 / days) ** 0.5))
        fckj = betta * self.fck
        return fckj

    def fcdj(self, j=28):
        return self.fckj(j) / self.yc

    def ecij(self, days=28):
        return 5600 * (self.fckj(days) ** (1 / 2))


@dataclass
class ReinforcedConcrete:
    concrete: Concrete
    rebar: Rebar


# sections.py

@dataclass
class RectangularSection:
    base: int
    height: int
    alpha: int = 1.5  # ABNT NBR 6118: 2014, art. 17.3, p. 17.3.1

    def area(self):
        return self.base * self.height

    def area_effective(self):
        return (self.base * self.height) / 1.5  # todo: test value

    def inertia(self):
        return (self.base * self.height ** 3) / 12


@dataclass
class ReinforcedConcreteSection:
    material: ReinforcedConcrete
    geometry: RectangularSection
    cover: int

    def d(self):
        return self.geometry.height - self.cover

    def alpha_e(self):
        return self.material.rebar.Es / self.material.concrete.ecs()

    def mcr(self):
        return (self.geometry.alpha * (self.material.concrete.fctm() / 10) * self.geometry.inertia()) / \
               (self.geometry.height / 2)

    def x1(self):
        return (((self.geometry.base * self.geometry.height ** 2) / 2) + (
                self.alpha_e() - 1) * self.material.rebar.As * self.d()) / \
               (self.geometry.area() + (self.alpha_e() - 1) * self.material.rebar.As)

    def inertia1(self):
        return self.geometry.inertia() + self.geometry.area() * (self.x1() - self.geometry.height / 2) ** 2 + \
               (self.alpha_e() - 1) * self.material.rebar.As * (self.d() - self.x1()) ** 2

    def x2(self):
        a = self.geometry.base / 2
        b = (self.alpha_e() - 1) * self.material.rebar.As
        c = - (self.alpha_e() - 1) * self.material.rebar.As * self.d()
        coeff = [a, b, c]
        possible_values = np.roots(coeff)
        for val in possible_values:
            if 0 < val < self.geometry.height:
                return val
        return 0

    def inertia2(self):
        return (self.geometry.base * self.x2() ** 3) / 3 + \
               self.alpha_e() * self.material.rebar.As * (self.d() - self.x2()) ** 2

    def ea(self):
        return self.material.concrete.eci() * self.geometry.area()

    def ei1(self):
        return self.material.concrete.ecs() * self.inertia1()


# writer.py

def list_files(path):
    r = []
    for root, dirs, files in os.walk(path):
        for name in files:
            r.append(os.path.join(root, name))
    return r


def list_folders(path):
    r = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            r.append(os.path.join(root, name))
    return r


def get_max_deflections_output_df(output_path, beam_name):
    for folder in list_folders(output_path):
        print(folder)
        if folder == output_path + beam_name:
            print("yes!")
            max_displacement_df = pd.DataFrame()
            min_displacement_df = pd.DataFrame()
            for file in list_files(folder):
                df = pd.read_csv(file)
                maximum_displacement_row = df.loc[df['uy'].idxmax()].copy()
                maximum_displacement_row['name'] = os.path.basename(file)
                max_displacement_df = pd.concat([max_displacement_df, maximum_displacement_row.to_frame().T], ignore_index=True)

                minimum_displacement_row = df.loc[df['uy'].idxmin()].copy()
                minimum_displacement_row['name'] = os.path.basename(file)
                min_displacement_df = pd.concat([min_displacement_df, minimum_displacement_row.to_frame().T], ignore_index=True)
            for ele, row in max_displacement_df.iterrows():
                if row['cracked'] == 0:
                    max_displacement_df.at[ele, 'branson'] = row['uy']
                    max_displacement_df.at[ele, 'bischoff'] = row['uy']
            for ele, row in min_displacement_df.iterrows():
                if row['cracked'] == 0:
                    min_displacement_df.at[ele, 'branson'] = row['uy']
                    min_displacement_df.at[ele, 'bischoff'] = row['uy']
            return max_displacement_df, min_displacement_df
    return None


def write_outputs_to_csv(folder, beam):
    max_displacement_df = pd.DataFrame()
    min_displacement_df = pd.DataFrame()
    for file in list_files(folder):
        df = pd.read_csv(file)
        maximum_displacement_row = df.loc[df['uy'].idxmax()].copy()
        maximum_displacement_row.loc['name'] = os.path.basename(file)
        maximum_displacement_row.loc['q_load'] = os.path.basename(file).split('-')[-1][:-4]
        max_displacement_df = pd.concat([max_displacement_df, maximum_displacement_row.to_frame().T], ignore_index=True)
        minimum_displacement_row = df.loc[df['uy'].idxmin()].copy()
        minimum_displacement_row.loc['name'] = os.path.basename(file)
        minimum_displacement_row.loc['q_load'] = os.path.basename(file).split('-')[-1][:-4]
        min_displacement_df = pd.concat([min_displacement_df, minimum_displacement_row.to_frame().T], ignore_index=True)
    max_displacement_df.sort_values('q_load', axis=0, ascending=True, inplace=True)
    max_displacement_df.to_csv(folder + "deslocamento_maximo.csv")
    min_displacement_df.sort_values('q_load', axis=0, ascending=True, inplace=True)
    min_displacement_df.to_csv(folder + "deslocamento_minimo.csv")
    with open(f"{folder + beam.name}.txt", '+w') as f:
        f.write(beam.input_describe())
    

# beam.py


OUTPUT_PATH = 'análise/'
BEAM_DATA_FILE = 'output_analysis/beams_input_data.csv'


class Beam:
    def __init__(self,
                 name: str,
                 nodes: list,
                 sups: list,
                 section: ReinforcedConcreteSection,
                 q_load: float,
                 discretization: int,
                 load_step: float,
                 elements: dict = None,
                 data: pd.DataFrame = None,
                 analysis: BeamAnalysis = None):
        self.name = name
        self.nodes = nodes
        self.supports = sups
        self.section = section
        self.q_load = q_load
        self.discretization = discretization
        self.load_step = load_step
        self.ss = SystemElements()
        self.total_length = np.linalg.norm(np.array(nodes[-1].coords) - [0.0, 0.0])
        self.solved = False
        self.branson_deflection = 0
        self.bischoff_deflection = 0
        self.analysis = analysis
        if elements is None:
            elements = {}
        self.elements = elements
        if data is None:
            data = pd.DataFrame()
        self.data = data

    def save_analysis_data(self):
        dataframes = []
        for case in self.analysis.cases.values():
            df = pd.DataFrame()
            if not self.solved:
                return dataframes
            df['element'] = [x for x in case.bars.keys()]
            bars = case.bars.values()
            df['ei'] = [el.EI for el in bars]
            df['ea'] = [el.EA for el in bars]
            df['shear'] = [el.nodes[0].V for el in bars]
            df['shear_2'] = [el.nodes[1].V for el in bars]
            df['moment'] = [el.nodes[0].M for el in bars]
            df['moment_2'] = [el.nodes[1].M for el in bars]
            df['uy'] = [el.nodes[0].uy * 10 for el in bars]
            df['uy_2'] = [el.nodes[1].uy * 10 for el in bars]
            df['phi'] = [el.nodes[0].phi for el in bars]
            df['phi_2'] = [el.nodes[1].phi for el in bars]
            df['cracked'] = [x in self.cracked_elements().keys() for x in df['element']]
            df.to_csv("vigas/V001/etapas/" + self.name + '-{:.2f}.csv'.format(case.load * 100), index=False)
            dataframes.append(df)
        write_outputs_to_csv("vigas/V001/", self)
        return dataframes

    def write_analysis_to_csv(self, folder=""):
        self.get_analysis_dataframe().to_csv(OUTPUT_PATH + folder + self.name + '.csv', index=False)

    def branson_inertia(self, actual_moment):
        ief = branson_equation(abs(self.section.mcr()),
                               abs(actual_moment),
                               self.section.inertia1(),
                               self.section.inertia2(),
                               n=4)
        return ief

    def ei_br(self, actual_moment):
        effective_stiffness = self.branson_inertia(actual_moment) * self.section.material.concrete.ecs()
        return effective_stiffness

    def add_elements(self):
        for ele in self.nodes:
            if ele == Node(0, 0):
                continue
            self.ss.add_element(location=[ele.coords], EA=self.section.ea(), EI=self.section.ei1())
        for sup in self.supports:
            n = self.ss.find_node_id(Vertex(sup.node.coords))
            if not any(self.ss.supports_fixed):
                self.ss.add_support_hinged(node_id=n)
                continue
            self.ss.add_support_roll(node_id=n)

    def add_elements_from_dict(self):
        for k, v in self.elements.items():
            if k == 0:
                continue
            self.ss.add_element(location=[v.coords()], EA=v.EA, EI=v.EI)
        for sup in self.supports:
            if not any(self.ss.supports_fixed):
                n = self.ss.find_node_id(Vertex(sup.node.coords))
                self.ss.add_support_hinged(node_id=n)
                continue
            n = self.ss.find_node_id(Vertex(sup.node.coords))
            self.ss.add_support_roll(node_id=n)

    def cracked_elements(self):
        elements = {}
        if not self.solved:
            return elements
        for k, v in self.ss.element_map.items():
            if (abs(v.bending_moment[0]) > abs(self.section.mcr())) or (
                    abs(v.bending_moment[-1]) > abs(self.section.mcr())):
                start_node = Node(*v.vertex_1.coordinates)
                end_node = Node(*v.vertex_2.coordinates)
                moment_in_act = max(abs(v.bending_moment[0]), abs(v.bending_moment[-1]))
                elements[k] = Bar(start_node, end_node, int(self.section.ea()),
                                  ceil(self.ei_br(moment_in_act) / 100) * 100)
        return elements

    def get_max_deflection_value(self):
        if not self.solved:
            return -1
        df = self.get_analysis_dataframe()
        deflection_value_series = df.loc[abs(df['uy']) == max(abs(df['uy'])), 'uy']
        deflection_value = deflection_value_series.to_list()[-1]
        return deflection_value

    def solve_beam(self, load=0):
        if load == 0:
            load = self.q_load
        for k in self.ss.element_map.keys():
            self.ss.q_load(q=load, element_id=k)
        self.ss.solve()
        self.solved = True
        return self.solved

    def create_analysis_bars_and_nodes(self, load_case_object):
        df = self.get_analysis_dataframe()
        for ele, row in df.iterrows():
            new_node_1 = NodeObject(
                int(row['element']),
                float(row['shear']),
                float(row['moment']),
                float(row['uy']),
                float(row['phi'])
            )
            new_node_2 = NodeObject(
                int(row['element']) + 1,
                float(row['shear_2']),
                float(row['moment_2']),
                float(row['uy_2']),
                float(row['phi_2'])
            )
            new_nodes = {0: new_node_1, 1: new_node_2}

            new_bar = BarObject(row['element'], row['ei'], row['ea'], row['cracked'], new_nodes)
            load_case_object.bars[new_bar.id] = new_bar
        return load_case_object

    def input_describe(self):
        description = "  Dados da Viga {}\n\n".format(self.name) + \
                      "  Fck: {} MPa\n".format(self.section.material.concrete.fck) + \
                      "  Fct: {:.2f} MPa\n".format(self.section.material.concrete.fctm()) + \
                      "  Ic: {:.2f} cm4\n".format(self.section.geometry.inertia()) + \
                      "  Eci: {:.2f} MPa\n".format(self.section.material.concrete.eci()) + \
                      "  Ecs: {:.2f} MPa\n".format(self.section.material.concrete.ecs()) + \
                      "  v: {} \n".format(self.section.material.concrete.v) + \
                      "  Alfa: {} /ºC\n\n".format(self.section.material.concrete.alpha) + \
                      "  b: {} cm\n".format(self.section.geometry.base) + \
                      "  h: {} cm\n".format(self.section.geometry.height) + \
                      "  Area: {} cm²\n".format(self.section.geometry.area()) + \
                      "  As: {} cm²\n".format(self.section.material.rebar.As) + \
                      "  Taxa Arm.: {:.3f} %\n".format(self.section.material.rebar.As /
                                                       self.section.geometry.area() * 100) + \
                      "  cobrimento: {} cm\n".format(self.section.cover) + \
                      "  d: {} cm\n".format(self.section.geometry.height - self.section.cover) + \
                      "  Mr: {} kN.cm\n\n".format(int(self.section.mcr())) + \
                      "  Carregamento: {:.1f} kN/m\n".format(abs(self.q_load) * 100) + \
                      "  Vão: {:.1f} m\n".format(self.total_length / 100) + \
                      "  alpha_e: {:.4f} \n".format(self.section.alpha_e()) + \
                      "  x1: {:.2f} cm\n".format(self.section.x1()) + \
                      "  x2: {:.2f} cm\n ".format(self.section.x2()) + \
                      "  x2/d: {:.2f}\n ".format(self.section.x2() / self.section.d()) + \
                      "  I1: {:.2f} cm4\n ".format(self.section.inertia1()) + \
                      "  I2: {:.2f} cm4\n\n ".format(self.section.inertia2()) + \
                      "  Nº barras: {:.0f}\n".format(self.total_length / self.discretization) + \
                      "  Discretização: {} cm\n".format(self.discretization) + \
                      "  Etapas de carga: {:.0f}\n".format(abs(self.q_load * 100) / self.load_step) + \
                      "  Inc. Carga: {} kN/m\n".format(self.load_step)
        return description

    def get_analysis_dataframe(self):
        df = pd.DataFrame()
        if not self.solved:
            return df
        df['element'] = [x for x in self.ss.element_map.keys()]
        values = self.ss.element_map.values()
        df['ei'] = [el.EI for el in values]
        df['ea'] = [el.EA for el in values]
        df['shear'] = [el.shear_force[0] for el in values]
        df['shear_2'] = [el.shear_force[-1] for el in values]
        df['moment'] = [el.bending_moment[0] for el in values]
        df['moment_2'] = [el.bending_moment[-1] for el in values]
        df['uy'] = [el.node_1.uz * 10 for el in values]
        df['uy_2'] = [el.node_2.uz for el in values]
        df['phi'] = [el.node_1.phi_y for el in values]
        df['phi_2'] = [el.node_2.phi_y for el in values]
        df['cracked'] = [x in self.cracked_elements().keys() for x in df['element']]
        return df


# deflections.py

def get_branson_deflection(length, ecs, mr, ma, inertia_1, inertia_2):
    return (5 * ma * length ** 2) / (48 * (ecs / 10) * branson_equation(mr, ma, inertia_1, inertia_2))


def branson_equation(mr, ma, inertia_1, inertia_2, n=3):
    effective_inertia = (mr / ma) ** n * inertia_1 + (1 - (mr / ma) ** n) * inertia_2
    return effective_inertia


def get_bischoff_deflection(length, ecs, mr, ma, inertia_1, inertia_2):
    return (5 * ma * length ** 2) / (48 * (ecs / 10) * bischoff_equation(mr, ma, inertia_1, inertia_2))


def bischoff_equation(mr, ma, inertia_1, inertia_2):
    effective_inertia = inertia_2 / (1 - (((mr / ma) ** 2) * (1 - (inertia_2 / inertia_1))))
    return effective_inertia


# elements.py

class Node:
    def __init__(self, xx, yy):
        self.xx = xx
        self.yy = yy
        self.coords = np.array([xx, yy])

    def __str__(self):
        return f"%d,%d" % (self.xx, self.yy)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.xx == other.xx and self.yy == other.yy


@dataclass
class Bar:
    node_a: Node
    node_b: Node
    EA: int
    EI: int

    def length(self):
        return np.linalg.norm(self.node_a.coords - self.node_b.coords)

    def coords(self):
        return self.node_b.coords


@dataclass
class Support:
    node: Node
    xx: bool
    yy: bool
    zz: bool


# solver.py

def solve_beam_incrementally(original_beam, load_step):
    beam_analysis = BeamAnalysis(original_beam.name, {})
    if original_beam.q_load < 0:
        load_step *= -0.01
    load = load_step
    already_cracked_elements = {}
    beam_elements = original_beam.elements
    beam = None
    new_name = ""
    while abs(load) <= round(abs(original_beam.q_load + load_step / 2), 4):
        new_load_case = LoadCaseObject(abs(load), {}, 0.0, 0.0)
        if new_name == original_beam.name + "L" + str(load):
            new_name = new_name + "+1"
        else:
            new_name = original_beam.name + "L{:.5f}".format(load)
        beam = Beam(new_name,
                    original_beam.nodes,
                    original_beam.supports,
                    original_beam.section,
                    load,
                    original_beam.discretization,
                    original_beam.load_step,
                    beam_elements,
                    original_beam.data)
        beam.add_elements_from_dict()
        beam.solve_beam()
        ma = abs(beam.q_load * beam.total_length ** 2) / 8
        if (ma != 0) and (ma > beam.section.mcr()):
            beam.branson_deflection = get_branson_deflection(beam.total_length, beam.section.material.concrete.ecs(),
                                                             abs(beam.section.mcr()), abs(ma),
                                                             abs(beam.section.inertia1()), abs(beam.section.inertia2()))
            beam.bischoff_deflection = get_bischoff_deflection(beam.total_length, beam.section.material.concrete.ecs(),
                                                               abs(beam.section.mcr()), abs(ma),
                                                               abs(beam.section.inertia1()), abs(beam.section.inertia2()
                                                                                                 ))
            new_load_case.branson = beam.branson_deflection
            new_load_case.bischoff = beam.bischoff_deflection
        else:
            beam.branson_deflection = beam.get_max_deflection_value()
            beam.bischoff_deflection = beam.get_max_deflection_value()
            new_load_case.branson = beam.branson_deflection
            new_load_case.bischoff = beam.bischoff_deflection
        new_load_case = beam.create_analysis_bars_and_nodes(new_load_case)
        beam_analysis.cases[new_load_case.load] = new_load_case
        new_cracked_elements = beam.cracked_elements()
        if new_cracked_elements != {}:
            if already_cracked_elements == {}:
                already_cracked_elements = new_cracked_elements
            elif new_cracked_elements == already_cracked_elements:
                load += load_step
            elif len(new_cracked_elements) < len(already_cracked_elements):
                load += load_step
            else:
                if new_cracked_elements.keys() == already_cracked_elements.keys():
                    step = True
                    for k in new_cracked_elements.keys():
                        if new_cracked_elements[k].EI == already_cracked_elements[k].EI:
                            continue
                        else:
                            if new_cracked_elements[k].EI < already_cracked_elements[k].EI:
                                step = False
                    if step:
                        load += load_step
                already_cracked_elements.update(new_cracked_elements)
        else:
            load += load_step
        beam_elements.update(already_cracked_elements)
    if beam is not None:
        beam.name = original_beam.name
        beam.analysis = beam_analysis
        beam.solved = True
        return beam
    return None


# runner.py

def create_beam_from_dict(input_dict, discretization, load_step):
    section_geometry = RectangularSection(input_dict['b'], input_dict['h'])
    section_material = ReinforcedConcrete(Concrete(input_dict['fck'], input_dict['yc'], input_dict['gamma'], 1),
                                          Rebar(500, 1.15, 7850, 210000, input_dict['as'], input_dict["asl"]), )
    section = ReinforcedConcreteSection(section_material, section_geometry, input_dict['cover'])
    total_len = input_dict['l1'] + input_dict['l2']
    x = 0
    last_node = Node(0, 0)
    node_dict = {"0,0": last_node}
    nodes = [last_node, ]
    length_step = discretization
    elements_dict = {}
    while x < total_len:
        x += length_step
        new_node = Node(x, 0)
        elements_dict[x / length_step] = Bar(last_node, new_node, section.ea(), section.ei1())
        nodes.append(new_node)
        node_dict[str(new_node)] = new_node
        last_node = new_node
    if int(input_dict['l2']) == 0:
        supports = [Support(node_dict['0,0'], True, True, False),
                    Support(node_dict[f'%d,0' % input_dict['l1']], True, True, False)]
    else:
        supports = [Support(node_dict['0,0'], True, True, False),
                    Support(node_dict[f'%d,0' % input_dict['l1']], True, True, False),
                    Support(node_dict[f'%d,0' % int(input_dict['l2'] + input_dict['l1'])], True, True, False)]
    load = -input_dict['q1'] / 100
    beam_name = input_dict['name']
    b = Beam(beam_name, nodes, supports, section, load, discretization, load_step, elements=elements_dict,
             data=input_dict)
    return b


def run_beam_list(beams, load_step):
    solved_beams = []
    for b in beams:
        # print(b)
        b.add_elements()
        new_beam = solve_beam_incrementally(b, load_step)
        solved_beams.append(new_beam)
    return solved_beams


def run_from_app(input_dict):
    beam = create_beam_from_dict(input_dict, input_dict['discretization'], input_dict['load_step'])
    beam = run_beam_list([beam, ], input_dict['load_step'])[-1]
    return beam


# App

class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        try:
            title = kwargs.pop('title')
        except KeyError:
            title = 'Untitled'
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, title)
        self.container = tk.Frame(self)
        self.container.grid(row=0, column=0)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.beam_dict = {}
        self.frames = {}
        self.active_frame = None
        for F in FRAMES:
            frame = F(self.container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nesw")
        self.menubar = Menu(self)
        self.config(menu=self.menubar)
        self.show_frame("StartPage")
        self.solved = False

    def show_frame(self, target_frame):
        self.active_frame = self.frames[target_frame]
        self.active_frame.tkraise()


class Menu(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.add_command(label="Início", command=lambda: parent.show_frame("StartPage"))
        self.add_command(label="Resultados", command=lambda: parent.show_frame("OutputPage"))
        self.add_command(label="Sobre", command=lambda: parent.show_frame("AboutPage"))


class BaseFrame(tk.Frame):
    instances = []

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        BaseFrame.instances.append(self)


class AboutPage(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent)
        self.root = root
        row_counter = 0
        line_break = ttk.Label(self, text=' ')
        line_break.grid(row=row_counter, column=4)
        line_break.grid(row=row_counter, column=5)
        line_break.grid(row=row_counter, column=6)
        line_break.grid(row=row_counter, column=7)
        line_break.grid(row=row_counter, column=8)
        row_counter += 1
        label = ttk.Label(self, text="                        ", font=LARGE_FONT)
        label.grid(row=row_counter, column=0)
        label = ttk.Label(self, text="Aviso:", font=LARGE_FONT)
        label.grid(row=row_counter, column=2)
        row_counter += 1

        heading1 = ttk.Label(self, text="Este aplicativo foi desenvolvido como parte de um trabalho de conclusão")
        heading1.grid(row=2, column=2)
        heading2 = ttk.Label(self, text="de curso de engenharia civil pela Universidade Federal de São Carlos.")
        heading2.grid(row=3, column=2)
        heading3 = ttk.Label(self, text="Trata-se de uma calculadora de flechas imediatas, para vigas biapoiadas de")
        heading3.grid(row=4, column=2)
        heading4 = ttk.Label(self, text="concreto armado. O programa não verifica se as solicitações na viga ")
        heading4.grid(row=5, column=2)
        heading5 = ttk.Label(self, text="ultrapassam a resistência máxima do elemento estrutural. ")
        heading5.grid(row=6, column=2)
        heading5 = ttk.Label(self, text="Caso necessário, entre em contato via email, dev.pedrodeo@gmail.com. ")
        heading5.grid(row=7, column=2)
        heading5 = ttk.Label(self, text="https://repositorio.ufscar.br/handle/ufscar/16716. ")
        heading5.grid(row=8, column=2)
        ttk.Label(self, text=' ').grid(row=9, column=3)

        self.p1 = ttk.Label(self, text="Lista de símbolos", font=LARGE_FONT)
        self.p1.grid(row=10, column=2)
        ttk.Label(self, text="Fck - Resistência à compressão característica do concreto").grid(row=11, column=2)
        ttk.Label(self, text="As - Armadura positiva da viga").grid(row=12, column=2)
        ttk.Label(self, text="As' - Armadura negativa da viga").grid(row=13, column=2)
        ttk.Label(self, text="Fct - Resistência a tração do concreto").grid(row=14, column=2)
        ttk.Label(self, text="Ic - Momento de inércia da seção de concreto").grid(row=15, column=2)
        ttk.Label(self, text="Eci - Módulo de elasticidade do concreto").grid(row=16, column=2)
        ttk.Label(self, text="Ecs - Módulo de elasticidade secante do concreto").grid(row=17, column=2)
        ttk.Label(self, text="v - Coeficiente de Poisson").grid(row=18, column=2)
        ttk.Label(self, text="alfa - Coeficiente de dilatação térmica do concreto").grid(row=19, column=2)
        ttk.Label(self, text="b - Base da seção retangular").grid(row=20, column=2)
        ttk.Label(self, text="h - Altura da seção retangular").grid(row=21, column=2)
        ttk.Label(self, text="Taxa arm. - Taxa de armadura da seção").grid(row=22, column=2)
        ttk.Label(self, text="d - Altura útil da seção").grid(row=23, column=2)
        ttk.Label(self, text="Mr - Momento de fissuração").grid(row=24, column=2)
        ttk.Label(self, text="alpha_e - Relação entre os módulos de elasticidade do concreto e do aço").grid(row=25,
                                                                                                             column=2)
        ttk.Label(self, text="x1 - profundidade da linha neutra no estádio II").grid(row=26, column=2)
        ttk.Label(self, text="x2 - profundidade da linha neutra no estádio II").grid(row=27, column=2)
        ttk.Label(self, text="x2/d - proporção entre profundidade da L.N. e a altura útil da peça").grid(row=28,
                                                                                                         column=2)
        ttk.Label(self, text="I1 - Inércia da seção homogeneizada no estádio I").grid(row=29, column=2)
        ttk.Label(self, text="I2 - Inércia da seção homogeneizada no estádio II").grid(row=30, column=2)


class OutputPage(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent)
        self.root = root
        self.beam = None
        self.heading = ttk.Label(self, text="Resultados da análise")
        self.heading.grid(row=1, column=2)
        beam_option_label = ttk.Label(self, text="Viga")
        beam_option_label.grid(row=2, column=1)
        self.beam_var = tk.StringVar(self)
        self.beam_var.set("")  # default value
        self.beam_var_option = tk.OptionMenu(self, self.beam_var, "")
        self.beam_var_option.grid(row=2, column=2)
        button1 = ttk.Button(self, text="Momento/Deformação", command=lambda: self.plot_graph())
        button1.grid(row=4, column=1)
        button2 = ttk.Button(self, text="Momento Fletor", command=lambda: self.show_bending_moment())
        button2.grid(row=4, column=2)
        button3 = ttk.Button(self, text="Força Cortante", command=lambda: self.show_shear_forces())
        button3.grid(row=4, column=3)
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.figure.set_facecolor('grey')
        self.deflection = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=5, column=1, columnspan=3)
        self.beam_description = ttk.Label(self, text='')
        self.beam_description.grid(row=5, column=4, )
        self.footing = ttk.Label(self, text="")
        self.footing.grid(row=6, column=2)
        self.save_msg = ttk.Label(self,
                                  text="Ao salvar uma viga com nome já existente, os arquivos serão sobrescritos!")
        self.save_msg.grid(row=7, column=1, columnspan=3)
        save_button = ttk.Button(self, text="Salvar dados", command=lambda: self.save_beam_data())
        save_button.grid(row=7, column=4)
        self.last_footer = ttk.Label(self, text="")
        self.last_footer.grid(row=8, column=2)

    def save_beam_data(self):
        if self.beam:
            if not os.path.isdir("vigas/"):
                os.mkdir("vigas/")
            if os.path.isdir("vigas/" + self.beam.name):
                rmtree("vigas/" + self.beam.name)
            os.mkdir("vigas/" + self.beam.name)
            os.mkdir("vigas/" + self.beam.name + "/etapas/")
            self.beam.save_analysis_data()
        self.save_msg.configure(text="Arquivos salvos em: .../vigas/{}/".format(self.beam.name))

    def show_shear_forces(self):
        if self.beam:
            self.figure.clear()
            df = self.beam.analysis.get_shear_diagram_dataframe(self.beam.total_length)
            self.beam_description.configure(text="Cortante Máxima: {:.2f} kN".format(abs(min(df['shear']))))
            self.deflection = self.figure.add_subplot(111)
            self.deflection.plot(df['length'], df['shear'], color='b')
            self.deflection.plot(df['length'], [0 for _ in df['length']], color='r')
            contour1 = [[0, 0], [0, df.at[0, 'shear']]]
            contour2 = [[self.beam.total_length, self.beam.total_length], [0, -df.at[0, 'shear']]]
            self.deflection.plot(contour1[0], contour1[1], 'b')
            self.deflection.plot(contour2[0], contour2[1], 'b')
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=5, column=1, columnspan=3)

    def show_bending_moment(self):
        if self.beam:
            self.figure.clear()
            df = self.beam.analysis.get_bending_diagram_dataframe(self.beam.total_length)
            self.beam_description.configure(text="Mmáx: {:.2f} kN.cm".format(abs(min(df['moment']))))
            self.deflection = self.figure.add_subplot(111)
            self.deflection.plot(df['length'], df['moment'], color='b')
            self.deflection.plot(df['length'], [0 for _ in df['length']], color='r')
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=5, column=1, columnspan=3)

    def refresh_options(self, time_now):
        self.last_footer.configure(text="Tempo de execução: {:.2f} segundos.".format(time_now))
        if self.beam is None:
            return
        choice = self.beam.name
        self.beam_var_option['menu'].add_command(label=choice, command=tk._setit(self.beam_var, choice))
        self.beam_var.set(choice)

    def plot_graph(self):
        if self.beam:
            self.figure.clear()
            self.beam_description.configure(text=self.beam.input_describe())
            df = self.beam.analysis.get_graph_dataframe()
            self.deflection = self.figure.add_subplot(111)
            self.deflection.plot(df['branson'], df['moment'], color='r')
            self.deflection.plot(df['bischoff'], df['moment'], color='g')
            self.deflection.plot(df['mef'], df['moment'], color='b')
            self.deflection.set_xlabel('Flecha')
            self.deflection.set_ylabel('Momento atuante')
            max_moment = max(abs(df['moment']))
            max_deflection = max(max(abs(df['bischoff'])), max(abs(df['mef'])))
            grid_lines = [[[0, 0], [0, max_moment + 1000]], ]
            d = 0
            while d < max_deflection:
                d += 0.25
                grid_lines.append([[d, d], [0, max_moment + 1000]])
            d += 0.25
            m = 0
            while m < max_moment:
                grid_lines.append([[0, d], [m, m]])
                m += 1000
            grid_lines.append([[0, d], [m, m]])
            for g in grid_lines:
                self.deflection.plot(g[0], g[1], color='grey', linewidth=0.25)
            legend_lines = [Line2D([0], [0], color='r', lw=4),
                            Line2D([0], [0], color='g', lw=4),
                            Line2D([0], [0], color='b', lw=4), ]
            self.deflection.legend(legend_lines, ['Branson', 'Bischoff', 'MEF-Branson'], loc='lower right')
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=5, column=1, columnspan=3)
            self.footing.configure(text='Flecha imediata máxima calculada por cada método:\n' +
                                        'Branson: {:.4f} cm\n'.format(max(abs(df['branson']))) +
                                        'Bischoff: {:.4f} cm\n'.format(max(abs(df['bischoff']))) +
                                        'MEF-Branson: {:.4f} cm\n'.format(max(abs(df['mef']))))


class StartPage(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent)
        self.root = root
        row_counter = 0
        line_break = ttk.Label(self, text=' ')
        line_break.grid(row=row_counter, column=4)
        line_break.grid(row=row_counter, column=5)
        line_break.grid(row=row_counter, column=6)
        line_break.grid(row=row_counter, column=7)
        line_break.grid(row=row_counter, column=8)
        row_counter += 1
        label = ttk.Label(self, text="                        ", font=LARGE_FONT)
        label.grid(row=row_counter, column=0)
        label = ttk.Label(self, text="Entre os dados abaixo", font=LARGE_FONT)
        label.grid(row=row_counter, column=2)
        row_counter += 1
        line_break = ttk.Label(self, text=' ')
        line_break.grid(row=row_counter, column=1)
        row_counter += 1
        name_label = ttk.Label(self, text="Nome")
        name_label.grid(row=row_counter, column=2)
        self.beam_name_field = ttk.Entry(self, textvariable="name")
        self.beam_name_field.insert(0, "V001")
        self.beam_name_field.grid(row=row_counter, column=3)
        row_counter += 1
        line_break = ttk.Label(self, text=' ')
        line_break.grid(row=row_counter, column=1)
        row_counter += 1
        ttk.Label(self, text="Material").grid(row=row_counter, column=1)
        ttk.Label(self, text="Fck").grid(row=row_counter, column=2)
        self.fck_var = tk.StringVar(self)
        self.fck_var.set("50")
        fck_widget = tk.OptionMenu(self, self.fck_var, "20", "30", "35", "40", "45", "50")
        fck_widget.grid(row=row_counter, column=3)
        ttk.Label(self, text="MPa").grid(row=row_counter, column=4)
        row_counter += 1
        line_break = ttk.Label(self, text=' ')
        line_break.grid(row=row_counter, column=1)
        row_counter += 1
        section_label = ttk.Label(self, text="Seção")
        section_label.grid(row=row_counter, column=1)
        section_x_label = ttk.Label(self, text="Base")
        section_x_label.grid(row=row_counter, column=2)
        self.section_x_field = ttk.Entry(self, textvariable="b")
        self.section_x_field.insert(0, "15")
        self.section_x_field.grid(row=row_counter, column=3)
        ttk.Label(self, text="cm").grid(row=row_counter, column=4)
        row_counter += 1
        section_y_label = ttk.Label(self, text="Altura")
        section_y_label.grid(row=row_counter, column=2)
        self.section_y_field = ttk.Entry(self, textvariable="h")
        self.section_y_field.insert(0, "35")
        self.section_y_field.grid(row=row_counter, column=3)
        ttk.Label(self, text="cm").grid(row=row_counter, column=4)
        row_counter += 1
        line_break = ttk.Label(self, text=' ')
        line_break.grid(row=row_counter, column=1)
        row_counter += 1
        rebar_label = ttk.Label(self, text='Armadura')
        rebar_label.grid(row=row_counter, column=1)
        rebar_posarea_label = ttk.Label(self, text='As')
        rebar_posarea_label.grid(row=row_counter, column=2)
        self.rebar_posarea_field = ttk.Entry(self, textvariable="as")
        self.rebar_posarea_field.insert(0, "1.51")
        self.rebar_posarea_field.grid(row=row_counter, column=3)
        ttk.Label(self, text="cm²").grid(row=row_counter, column=4)
        row_counter += 1
        rebar_negarea_label = ttk.Label(self, text="As'")
        rebar_negarea_label.grid(row=row_counter, column=2)
        self.rebar_negarea_field = ttk.Entry(self, textvariable="asl")
        self.rebar_negarea_field.insert(0, "1.51")
        self.rebar_negarea_field.grid(row=row_counter, column=3)
        ttk.Label(self, text="cm²").grid(row=row_counter, column=4)
        row_counter += 1
        rebar_cover_label = ttk.Label(self, text="Cobrimento")
        rebar_cover_label.grid(row=row_counter, column=2)
        self.rebar_cover_field = ttk.Entry(self, textvariable="cover")
        self.rebar_cover_field.insert(0, "5")
        self.rebar_cover_field.grid(row=row_counter, column=3)
        ttk.Label(self, text="cm").grid(row=row_counter, column=4)
        row_counter += 1
        line_break = ttk.Label(self, text=' ')
        line_break.grid(row=row_counter, column=1)
        row_counter += 1
        geometry_label = ttk.Label(self, text='Geometria')
        geometry_label.grid(row=row_counter, column=1)
        gap_label = ttk.Label(self, text='Vão 1')
        gap_label.grid(row=row_counter, column=2)
        self.gap_field = ttk.Entry(self, textvariable="gap1")
        self.gap_field.insert(0, "500")
        self.gap_field.grid(row=row_counter, column=3)
        ttk.Label(self, text="cm").grid(row=row_counter, column=4)
        row_counter += 1
        bar_count_label = ttk.Label(self, text='Número de barras')
        bar_count_label.grid(row=row_counter, column=2)
        self.bar_count_var = tk.StringVar(self)
        self.bar_count_var.set(20)
        bar_count_widget = tk.OptionMenu(self, self.bar_count_var, 4, 10, 20, 50, 100, 200)
        bar_count_widget.grid(row=row_counter, column=3)
        ttk.Label(self, text="nº barras").grid(row=row_counter, column=4)
        row_counter += 1
        line_break = ttk.Label(self, text=' ')
        line_break.grid(row=row_counter, column=1)
        row_counter += 1
        loads_label = ttk.Label(self, text='Carregamento')
        loads_label.grid(row=row_counter, column=1)
        load_type_label = ttk.Label(self, text='Tipo de Carregamento')
        load_type_label.grid(row=row_counter, column=2)
        self.load_type_var = tk.StringVar(self)
        self.load_type_var.set("Distribuído")
        load_type_widget = tk.OptionMenu(self, self.load_type_var, "Distribuído")
        load_type_widget.grid(row=row_counter, column=3)
        row_counter += 1
        load1_label = ttk.Label(self, text='Carga')
        load1_label.grid(row=row_counter, column=2)
        self.load1_field = ttk.Entry(self, textvariable="qload1")
        self.load1_field.insert(0, "14")
        self.load1_field.grid(row=row_counter, column=3)
        ttk.Label(self, text="KN/m").grid(row=row_counter, column=4)
        row_counter += 1
        ttk.Label(self, text="Etapas de carga").grid(row=row_counter, column=2)
        self.steps_var = tk.StringVar(self)
        self.steps_var.set("25")
        steps_widget = tk.OptionMenu(self, self.steps_var, "5", "10", "25", "50", "100")
        steps_widget.grid(row=row_counter, column=3)
        row_counter += 1
        line_break = ttk.Label(self, text=' ')
        line_break.grid(row=row_counter, column=1)
        row_counter += 1
        button = ttk.Button(self, text="Calcular", command=lambda: self.solve_beam())
        button.grid(row=row_counter, column=2)

    def get_field_values(self):
        disc = float(self.gap_field.get()) / int(self.bar_count_var.get())
        step = float(self.load1_field.get()) / int(self.steps_var.get())
        values = {'name': self.beam_name_field.get(),
                  'fck': int(self.fck_var.get()),
                  'yc': 1.4,
                  'gamma': 2500,
                  'v': 0.2,
                  'alpha': 1e-5,
                  'b': int(self.section_x_field.get()),
                  'h': int(self.section_y_field.get()),
                  'as': float(self.rebar_posarea_field.get()),
                  "asl": float(self.rebar_negarea_field.get()),
                  'cover': float(self.rebar_cover_field.get()),
                  'l1': int(self.gap_field.get()),
                  'l2': 0,
                  'q1': int(self.load1_field.get()),
                  'load_step': float(step),
                  'discretization': int(disc),
                  }
        return values

    def solve_beam(self):
        self.root.show_frame("OutputPage")
        start = time.time()
        beam = run_from_app(self.get_field_values())
        elapsed_time = time.time() - start
        self.root.solved = True
        self.root.frames["OutputPage"].beam = beam
        self.root.frames["OutputPage"].refresh_options(elapsed_time)
        self.root.frames["OutputPage"].plot_graph()


FRAMES = [StartPage, OutputPage, AboutPage]

if __name__ == "__main__":
    app = Application(title='ImediataVigas')
    app.geometry(RESOLUTION)
    app.mainloop()
