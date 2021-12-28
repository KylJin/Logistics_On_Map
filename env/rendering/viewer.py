import sys
import os.path as osp
import pygame
from pygame.locals import *
import igraph
import numpy as np
from scipy.spatial import distance
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, rgb2hex

resource_path = osp.join(osp.dirname(__file__), 'resources')

# NOTE: FPS*SPD应为24的倍数，否则可能导致货车到达终点时偏移仓库图标中心
FPS = 60  # Frame Per Second，帧率，即每秒播放的帧数
SPD = 4  # Second Per Day，游戏中每天所占的秒数
prov_map = {
    0: "北京", 1: "天津", 2: "河北", 3: "山西", 4: "内蒙古", 5: "辽宁", 6: "吉林", 7: "黑龙江",
    8: "上海", 9: "江苏", 10: "浙江", 11: "安徽", 12: "福建", 13: "江西", 14: "山东", 15: "河南",
    16: "湖北", 17: "湖南", 18: "广东", 19: "广西", 20: "海南", 21: "重庆", 22: "四川", 23: "贵州",
    24: "云南", 25: "西藏", 26: "陕西", 27: "甘肃", 28: "青海", 29: "宁夏", 30: "新疆"
}


class Viewer(object):
    def __init__(self, width, height, network_data):
        self.width = width
        self.height = height
        self.v_radius = 42
        self.n_vertex = network_data['n_vertex']
        self.pd_gaps = network_data['pd_gaps']  # 每个节点生产量和平均消耗量之间的差距
        self.connections = network_data['connections']
        self.trans_times = network_data['trans_times']
        if network_data['is_abs_coords']:  # 若提供了绝对坐标，则直接采用
            self.v_coords = network_data['v_coords']
        else:  # 若未提供坐标，或是相对坐标，则自动计算坐标
            self.v_coords = self._spread_vertex(network_data['v_coords'])

        pygame.init()
        pygame.display.set_caption("Simple Logistics Simulator")
        self.screen = pygame.display.set_mode([width, height])
        self.screen.fill("white")
        self.FPSClock = pygame.time.Clock()
        self.is_paused = False

        self.font1 = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 24)
        self.font2 = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 18)
        self.font3 = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 14)
        self.p_img = pygame.image.load(osp.join(resource_path, "img/produce.png")).convert_alpha()
        self.p_img = pygame.transform.scale(self.p_img, (16, 16))
        self.d_img = pygame.image.load(osp.join(resource_path, "img/demand.png")).convert_alpha()
        self.d_img = pygame.transform.scale(self.d_img, (14, 14))
        self.pause = pygame.image.load(osp.join(resource_path, "img/pause.png")).convert_alpha()
        self.play = pygame.image.load(osp.join(resource_path, "img/play.png")).convert_alpha()
        # 设置暂停按钮位置
        self.pause_rect = self.pause.get_rect()
        self.pause_rect.right, self.pause_rect.top = self.width - 10, 10

        self.background = self.init_background()
        self.warehouses = self.init_warehouses()
        self.trucks = self.init_trucks()

    def init_background(self):
        # 导入背景地图图像
        bg_img = pygame.image.load(osp.join(resource_path, "img/china_map.png")).convert_alpha()
        self.screen.blit(bg_img, (0, 0))

        # 绘制道路
        drawn_roads = []
        for i in range(self.n_vertex):
            start = self.v_coords[i]
            for j in self.connections[i]:
                if (j, i) in drawn_roads:
                    continue
                end = self.v_coords[j]
                self._draw_road(start, end, width=8,
                                border_color=(252, 122, 90), fill_color=(255, 172, 77))
                drawn_roads.append((i, j))
        # 加入固定的提示
        self.add_notation()

        # 保存当前初始化的背景，便于后续刷新时使用
        background = self.screen.copy()
        return background

    def init_warehouses(self):
        warehouse_list = []
        norm = Normalize(vmin=min(self.pd_gaps) - 200,
                         vmax=max(self.pd_gaps) + 200)  # 数值映射范围（略微扩大）
        color_map = get_cmap('RdYlGn')  # 颜色映射表
        for i in range(self.n_vertex):
            rgb = color_map(norm(self.pd_gaps[i]))[:3]
            color = pygame.Color(rgb2hex(rgb))
            warehouse = Warehouse(i, self.v_coords[i], color)
            warehouse_list.append(warehouse)

        return warehouse_list

    def init_trucks(self):
        trucks_list = []
        for i in range(self.n_vertex):
            start = self.v_coords[i]
            trucks = []
            for j, time in zip(self.connections[i], self.trans_times[i]):
                end = self.v_coords[j]
                truck = Truck((i, j), start, end, time)
                trucks.append(truck)
            trucks_list.append(trucks)

        return trucks_list

    def _spread_vertex(self, v_coords=None):
        if not v_coords:  # 若没有指定相对坐标，则随机将节点分布到画布上
            g = igraph.Graph()
            g.add_vertices(self.n_vertex)
            for i in range(self.n_vertex):
                for j in self.connections[i]:
                    g.add_edge(i, j)
            layout = g.layout_kamada_kawai()
            layout_coords = np.array(layout.coords).T
        else:  # 否则使用地图数据中指定的节点相对坐标
            layout_coords = np.array(v_coords).T

        # 将layout的坐标原点对齐到左上角
        layout_coords[0] = layout_coords[0] - layout_coords[0].min()
        layout_coords[1] = layout_coords[1] - layout_coords[1].min()

        # 将layout的坐标映射到画布坐标，并将图形整体居中
        stretch_rate = min((self.width - 2 * self.v_radius - 240) / layout_coords[0].max(),
                           (self.height - 2 * self.v_radius - 60) / layout_coords[1].max())
        # x方向左侧留出200，用于信息显示
        margin_x = (self.width - layout_coords[0].max() * stretch_rate) // 2 + 90
        margin_y = (self.height - layout_coords[1].max() * stretch_rate) // 2
        vertex_coord = []
        for i in range(self.n_vertex):
            x = margin_x + int(layout_coords[0, i] * stretch_rate)
            y = margin_y + int(layout_coords[1, i] * stretch_rate)
            vertex_coord.append((x, y))

        return vertex_coord

    def _draw_road(self, start, end, width, border_color=(0, 0, 0), fill_color=None):
        length = distance.euclidean(start, end)
        sin = (end[1] - start[1]) / length
        cos = (end[0] - start[0]) / length

        vertex = lambda e1, e2: (
            start[0] + (e1 * length * cos + e2 * width * sin) / 2,
            start[1] + (e1 * length * sin - e2 * width * cos) / 2
        )
        vertices = [vertex(*e) for e in [(0, -1), (0, 1), (2, 1), (2, -1)]]

        if not fill_color:
            pygame.draw.polygon(self.screen, border_color, vertices, width=3)
        else:
            pygame.draw.polygon(self.screen, fill_color, vertices, width=0)
            pygame.draw.polygon(self.screen, border_color, vertices, width=2)

    def add_notation(self):
        text1 = self.font3.render("黑:库存量", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(text1, (18, 65))

        text2 = self.font3.render(":生产量", True, (35, 138, 32), (255, 255, 255))
        self.screen.blit(text2, (32, 85))
        self.screen.blit(self.p_img, (17, 85))

        text3 = self.font3.render(":消耗量", True, (251, 45, 45), (255, 255, 255))
        self.screen.blit(text3, (32, 105))
        self.screen.blit(self.d_img, (17, 105))

        text4 = self.font3.render("蓝:节点奖赏", True, (12, 140, 210), (255, 255, 255))
        self.screen.blit(text4, (18, 125))

    def update(self, render_data):
        day = render_data['day']
        storages = render_data['storages']  # list per vertex
        productions = render_data['productions']  # list per vertex
        demands = render_data['demands']  # list per vertex
        total_reward = render_data['total_reward']
        single_rewards = render_data['single_rewards']
        actions = render_data['actions']  # matrix per vertex

        self.screen.blit(self.background, (0, 0))
        day_text = self.font1.render(f"第{day}天", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(day_text, (18, 10))
        r_text = self.font2.render(f"累计奖赏:{round(total_reward, 2)}", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(r_text, (18, 40))

        # 绘制每个仓库节点
        for warehouse, s, p, d, r in zip(self.warehouses, storages, productions, demands, single_rewards):
            warehouse.update(s, p, d, r)
            warehouse.draw(self.screen)

        # 绘制每个action对应的货车
        for i in range(self.n_vertex):
            for truck, action in zip(self.trucks[i], actions[i]):
                truck.update(action)
                truck.draw(self.screen)

    def _check_click(self, pos):
        # 检测是否点到暂停按钮
        if self.pause_rect.collidepoint(pos):
            self.is_paused = not self.is_paused
            return

        if self.is_paused:
            # 检测是否点击到仓库节点
            for warehouse in self.warehouses:
                details = warehouse.click(pos)
                if details:
                    height = details.get_height()
                    self.screen.blit(details, (0, self.height - height))
                    return

            # 检测是否点击到货车
            for i in range(self.n_vertex):
                for truck in self.trucks[i]:
                    details = truck.click(pos)
                    if details:
                        height = details.get_height()
                        self.screen.blit(details, (0, self.height - height))
                        return

    def render(self, render_data):
        current_frame = 0
        while current_frame < FPS * SPD:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    pressed_array = pygame.mouse.get_pressed(3)
                    if pressed_array[0]:
                        pos = pygame.mouse.get_pos()
                        self._check_click(pos)

            if not self.is_paused:
                self.update(render_data)
                self.screen.blit(self.pause, self.pause_rect)
                current_frame += 1
            else:
                self.screen.fill("white", self.pause_rect)
                self.screen.blit(self.play, self.pause_rect)

            self.FPSClock.tick(FPS)
            pygame.display.update()


class Warehouse(object):
    def __init__(self, key, pos, color=(0, 0, 0), radius=10):
        self.key = key
        self.rect = Rect(0, 0, 2 * radius, 2 * radius)
        self.rect.center = pos
        self.radius = radius
        self.color = color
        self.fill_color = self._lighten_color(color, alpha=0.2)

        self.font = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 14)
        self.p_img = pygame.image.load(osp.join(resource_path, "img/produce.png")).convert_alpha()
        self.p_img = pygame.transform.scale(self.p_img, (16, 16))
        self.d_img = pygame.image.load(osp.join(resource_path, "img/demand.png")).convert_alpha()
        self.d_img = pygame.transform.scale(self.d_img, (14, 14))

        self.storage = None
        self.production = None
        self.demand = None
        self.reward = None
        self.details = pygame.Surface((200, 200))

    def update(self, storage, production, demand, reward):
        self.storage = storage
        self.production = production
        self.demand = demand
        self.reward = reward

    def draw(self, screen):
        pygame.draw.circle(screen, self.fill_color, self.rect.center, self.radius, width=0)
        pygame.draw.circle(screen, self.color, self.rect.center, self.radius, width=2)

    def click(self, pos):
        if not self.rect.collidepoint(pos):
            return None
        white = (255, 255, 255)
        self.details.fill(white)

        key_text = self.font.render(f"省份:{prov_map[self.key]}", True, (44, 44, 44), white)
        self.details.blit(key_text, (18, 82))

        s_str = [str(round(s, 2)) for s in self.storage]
        s_text = self.font.render(f"库存:{','.join(s_str)}", True, (44, 44, 44), white)
        self.details.blit(s_text, (18, 104))

        p_str = [str(p) for p in self.production]
        p_text = self.font.render(f"生产:{','.join(p_str)}", True, (35, 138, 32), white)
        self.details.blit(p_text, (18, 126))

        d_str = [str(d) for d in self.demand]
        d_text = self.font.render(f"消耗:{','.join(d_str)}", True, (251, 45, 45), white)
        self.details.blit(d_text, (18, 148))

        r_text = self.font.render(f"奖赏:{round(self.reward, 2)}", True, (12, 140, 210), white)
        self.details.blit(r_text, (18, 170))

        return self.details

    @staticmethod
    def _lighten_color(color, alpha=0.1):
        r = alpha * color.r + (1 - alpha) * 255
        g = alpha * color.g + (1 - alpha) * 255
        b = alpha * color.b + (1 - alpha) * 255
        light_color = pygame.Color((r, g, b))
        return light_color


class Truck(object):
    def __init__(self, direction, start, end, trans_time, size=(20, 20)):
        self.dir = direction
        self.image = pygame.image.load(osp.join(resource_path, "img/truck.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, size)
        self.rect = self.image.get_rect()
        self.rect.center = start
        self.font = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 14)

        self.init_pos = (self.rect.x, self.rect.y)
        self.total_frame = trans_time * FPS * SPD // 24
        self.update_frame = 0

        speed_x = 24 * (end[0] - start[0]) / (trans_time * FPS * SPD)
        speed_y = 24 * (end[1] - start[1]) / (trans_time * FPS * SPD)
        self.speed = (speed_x, speed_y)

        self.action = None
        self.details = pygame.Surface((200, 200))

    def update(self, action):
        self.action = action
        if self.update_frame < self.total_frame:
            self.update_frame += 1
            self.rect.x = self.init_pos[0] + self.speed[0] * self.update_frame
            self.rect.y = self.init_pos[1] + self.speed[1] * self.update_frame
        else:
            self.update_frame += 1
            if self.update_frame >= FPS * SPD:
                self.update_frame = 0
                self.rect.topleft = self.init_pos

    def draw(self, screen):
        total = sum(self.action)
        if total <= 0:  # 若货车运输量为0，则不显示
            return
        # 当货车在道路上时才显示
        if 0 < self.update_frame < self.total_frame:
            screen.blit(self.image, self.rect)

    def click(self, pos):
        if not self.rect.collidepoint(pos):
            return None
        white = (255, 255, 255)
        self.details.fill(white)

        dir_text = self.font.render(f"方向:{prov_map[self.dir[0]]}->{prov_map[self.dir[1]]}",
                                    True, (44, 44, 44), white)
        self.details.blit(dir_text, (18, 148))

        a_str = [str(round(a, 2)) for a in self.action]
        a_text = self.font.render(f"运输:{','.join(a_str)}", True, (44, 44, 44), white)
        self.details.blit(a_text, (18, 170))

        return self.details
