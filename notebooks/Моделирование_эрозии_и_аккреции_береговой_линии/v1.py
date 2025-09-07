import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import pandas as pd
import folium
from IPython.display import display, HTML
import contextily as ctx
import random
import osmnx as ox
import requests
import io
from datetime import datetime, timedelta
from scipy.signal import savgol_filter

class DataCoastalModel:
    """Модель эрозии береговой линии с реальными данными"""

    def __init__(self, location_name: str, bbox: tuple, start_date: str = None):
        self.name = location_name
        self.bbox = bbox  # (west, south, east, north)
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now()
        self.current_date = self.start_date

        # Установка seed для воспроизводимости
        random.seed(42)
        np.random.seed(42)

        # Загрузка данных
        self._load_coastline_data()
        self._fetch_wave_data()
        self._generate_sediment_data()

        # Параметры модели
        self.erosion_rate = 0.0001  # Уменьшен для реалистичности
        self.accretion_rate = 0.00005  # Уменьшен для реалистичности
        self.geology_factor = 0.7

        # История изменений
        self.history = {
            'coastline': [],
            'dates': [],
            'wave_data': [],
            'sediment_data': []
        }

    def _load_coastline_data(self):
        """Загрузка данных береговой линии с высоким разрешением"""
        try:
            # Попытка загрузки из OpenStreetMap через Overpass API
            tags = {'natural': 'coastline'}
            gdf = ox.features_from_bbox(self.bbox[3], self.bbox[1], self.bbox[2], self.bbox[0], tags=tags)
            if gdf.empty:
                raise ValueError("Нет данных OSM для указанного региона")

            # Извлечение береговой линии
            lines = gdf[gdf.geom_type == 'LineString'].geometry
            if lines.empty:
                raise ValueError("Нет линий в данных OSM")
            if len(lines) > 1:
                self.coastline = MultiLineString(list(lines))
            else:
                self.coastline = lines.iloc[0]
            self.original_coastline = self.coastline
            print("Данные береговой линии загружены из OpenStreetMap")
            print(f"Тип геометрии: {self.coastline.geom_type}, Количество точек: "
                  f"{len(self.coastline.coords) if self.coastline.geom_type == 'LineString' else sum(len(line.coords) for line in self.coastline.geoms)}")

        except Exception as e:
            print(f"Ошибка загрузки данных OSM: {e}")
            print("Попытка загрузки из Natural Earth (1:10m)...")
            try:
                natural_earth_url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
                world = gpd.read_file(natural_earth_url)
                area = world.cx[self.bbox[0]:self.bbox[2], self.bbox[1]:self.bbox[3]]
                if area.empty:
                    raise ValueError("Нет данных для указанного региона")
                lines = area[area.geom_type == 'LineString'].geometry
                if lines.empty:
                    raise ValueError("Нет линий в данных Natural Earth")
                if len(lines) > 1:
                    self.coastline = MultiLineString(list(lines))
                else:
                    self.coastline = lines.iloc[0]
                self.original_coastline = self.coastline
                print("Данные загружены из Natural Earth (1:10m)")
                print(f"Тип геометрии: {self.coastline.geom_type}, Количество точек: "
                      f"{len(self.coastline.coords) if self.coastline.geom_type == 'LineString' else sum(len(line.coords) for line in self.coastline.geoms)}")
            except Exception as e:
                print(f"Ошибка загрузки из Natural Earth: {e}")
                print("Используются тестовые данные...")
                self._create_test_coastline()

    def _create_test_coastline(self):
        """Создание тестовой береговой линии"""
        x = np.linspace(self.bbox[0], self.bbox[2], 100)
        y = np.linspace(self.bbox[1], self.bbox[3], 100) + np.sin(x * 10) * 0.01
        self.coastline = LineString(np.column_stack((x, y)))
        self.original_coastline = self.coastline
        print("Создана тестовая береговая линия")
        print(f"Тип геометрии: {self.coastline.geom_type}, Количество точек: {len(self.coastline.coords)}")

    def _fetch_wave_data(self):
        """Генерация синтетических данных о волнах для Черного моря"""
        print("Используются синтетические данные волн для Черного моря...")
        dates = pd.date_range(start=self.start_date, periods=90, freq='D')
        self.wave_data = pd.DataFrame({
            'date': dates,
            'wave_height': np.random.normal(1.0, 0.3, len(dates)),
            'wave_direction': np.random.uniform(270, 360, len(dates))
        })
        print(f"Созданы синтетические данные волн: {len(self.wave_data)} строк")

    def _generate_sediment_data(self):
        """Генерация данных о наносах"""
        dates = pd.date_range(start=self.start_date, periods=90, freq='D')
        sediment = np.random.lognormal(mean=1.0, sigma=0.3, size=len(dates))
        self.sediment_data = pd.DataFrame({
            'date': dates,
            'sediment_supply': sediment
        })

    def _get_current_wave_conditions(self):
        """Получение текущих условий волнения"""
        mask = self.wave_data['date'] == self.current_date
        if mask.any():
            return self.wave_data[mask].iloc[0].to_dict()
        print(f"Нет данных волн для {self.current_date}, используются средние значения")
        return {
            'wave_height': np.mean(self.wave_data['wave_height']),
            'wave_direction': np.mean(self.wave_data['wave_direction'])
        }

    def _get_current_sediment_supply(self):
        """Получение текущего объема наносов"""
        mask = self.sediment_data['date'] == self.current_date
        if mask.any():
            return self.sediment_data[mask].iloc[0]['sediment_supply']
        return np.mean(self.sediment_data['sediment_supply'])

    def _calculate_single_line(self, line, wave_conditions):
        """Расчет изменений для одной линии с сглаживанием"""
        coords = np.array(line.coords)
        new_coords = coords.copy()

        for i in range(1, len(coords) - 1):
            dx = coords[i + 1][0] - coords[i - 1][0]
            dy = coords[i + 1][1] - coords[i - 1][1]
            length = np.sqrt(dx**2 + dy**2)
            if length == 0:
                continue
            normal = np.array([-dy, dx]) / length
            wave_rad = np.deg2rad(wave_conditions['wave_direction'])
            wave_vector = np.array([np.cos(wave_rad), np.sin(wave_rad)])
            cos_angle = np.dot(normal, wave_vector)

            if cos_angle > 0:
                erosion = (wave_conditions['wave_height'] - 0.5) * self.erosion_rate * cos_angle * self.geology_factor
                new_coords[i] -= normal * min(erosion, 0.0001)  # Строгое ограничение смещения
            else:
                if wave_conditions['wave_height'] < 0.5:
                    accretion = self.accretion_rate * self._get_current_sediment_supply() * abs(cos_angle)
                    new_coords[i] += normal * min(accretion, 0.0001)  # Строгое ограничение смещения

        # Сглаживание координат
        x_coords = savgol_filter(new_coords[:, 0], window_length=11, polyorder=2)
        y_coords = savgol_filter(new_coords[:, 1], window_length=11, polyorder=2)
        new_coords = np.column_stack((x_coords, y_coords))

        return LineString(new_coords)

    def _calculate_coastline_changes(self, wave_conditions):
        """Расчет изменений береговой линии с учетом нормали"""
        if self.coastline.is_empty:
            print("Береговая линия пуста, изменения не применяются")
            return self.coastline
        if self.coastline.geom_type == 'MultiLineString':
            new_geoms = [self._calculate_single_line(line, wave_conditions) for line in self.coastline.geoms]
            return MultiLineString(new_geoms)
        elif self.coastline.geom_type == 'LineString':
            return self._calculate_single_line(self.coastline, wave_conditions)
        print(f"Неподдерживаемый тип геометрии: {self.coastline.geom_type}")
        return self.coastline

    def simulate_day(self):
        """Моделирование одного дня"""
        wave_conditions = self._get_current_wave_conditions()
        sediment_supply = self._get_current_sediment_supply()
        self.coastline = self._calculate_coastline_changes(wave_conditions)
        self.current_date += timedelta(days=1)

        self.history['coastline'].append(self.coastline)
        self.history['dates'].append(self.current_date)
        self.history['wave_data'].append(wave_conditions)
        self.history['sediment_data'].append(sediment_supply)

    def simulate_period(self, days: int):
        """Моделирование на указанный период"""
        for _ in range(days):
            self.simulate_day()

    def plot_comparison(self):
        """Сравнение исходной и текущей береговых линий"""
        if self.coastline.is_empty or self.original_coastline.is_empty:
            print("Невозможно построить график: береговая линия пуста")
            return
        fig, ax = plt.subplots(figsize=(12, 8))
        # Применение проекции UTM для корректного масштаба
        orig_gs = gpd.GeoSeries(self.original_coastline, crs='EPSG:4326').to_crs('EPSG:32637')
        curr_gs = gpd.GeoSeries(self.coastline, crs='EPSG:4326').to_crs('EPSG:32637')
        orig_gs.plot(ax=ax, color='blue', label='Исходная')
        curr_gs.plot(ax=ax, color='red', label='Текущая')
        ax.set_title(f"Изменение береговой линии\n{self.name}")
        ax.legend()
        ctx.add_basemap(ax, crs='EPSG:32637', source=ctx.providers.OpenStreetMap.Mapnik)
        plt.show()

    def plot_interactive_map(self):
        """Интерактивная карта с Folium"""
        if self.coastline.is_empty or self.original_coastline.is_empty:
            print("Невозможно построить карту: береговая линия пуста")
            return None
        centroid = self.coastline.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14)
        orig_geojson = gpd.GeoSeries(self.original_coastline, crs='EPSG:4326').to_json()
        curr_geojson = gpd.GeoSeries(self.coastline, crs='EPSG:4326').to_json()
        folium.GeoJson(
            orig_geojson,
            style_function=lambda x: {'color': 'blue', 'weight': 2},
            tooltip="Исходная береговая линия"
        ).add_to(m)
        folium.GeoJson(
            curr_geojson,
            style_function=lambda x: {'color': 'red', 'weight': 3},
            tooltip="Текущая береговая линия"
        ).add_to(m)
        return m

    def get_erosion_stats(self):
        """Статистика изменений с учетом проекции"""
        if self.original_coastline.is_empty or self.coastline.is_empty:
            return {
                'original_length_km': 0,
                'current_length_km': 0,
                'length_change_m': 0,
                'erosion_rate_m_day': 0
            }
        # Применение проекции UTM для корректного расчета длины
        orig_gs = gpd.GeoSeries(self.original_coastline, crs='EPSG:4326').to_crs('EPSG:32637')
        curr_gs = gpd.GeoSeries(self.coastline, crs='EPSG:4326').to_crs('EPSG:32637')
        orig_length = orig_gs.length.iloc[0]
        curr_length = curr_gs.length.iloc[0]
        return {
            'original_length_km': orig_length / 1000,
            'current_length_km': curr_length / 1000,
            'length_change_m': curr_length - orig_length,
            'erosion_rate_m_day': (curr_length - orig_length) / max(1, len(self.history['coastline']))
        }

    def plot_erosion_animation(self):
        """Анимация изменений береговой линии"""
        if not self.history['coastline'] or any(geom.is_empty for geom in self.history['coastline']):
            print("Невозможно построить анимацию: история изменений пуста или содержит пустые геометрии")
            return None
        fig, ax = plt.subplots(figsize=(12, 8))

        def update(frame):
            ax.clear()
            orig_gs = gpd.GeoSeries(self.original_coastline, crs='EPSG:4326').to_crs('EPSG:32637')
            curr_gs = gpd.GeoSeries(self.history['coastline'][frame], crs='EPSG:4326').to_crs('EPSG:32637')
            orig_gs.plot(ax=ax, color='blue', label='Исходная')
            curr_gs.plot(ax=ax, color='red', label='Текущая')
            ax.set_title(f"{self.name}\n{self.history['dates'][frame].strftime('%Y-%m-%d')}")
            ax.legend()
            ctx.add_basemap(ax, crs='EPSG:32637', source=ctx.providers.OpenStreetMap.Mapnik)

        ani = FuncAnimation(
            fig, update, frames=len(self.history['coastline']),
            interval=300, repeat=False)
        plt.close()
        return ani

if __name__ == "__main__":
    # Использование с уточненными координатами Сочи
    model = DataCoastalModel(
        location_name="Сочи, Черноморское побережье",
        bbox=(39.7, 43.5, 39.8, 43.6),  # Уточненный bbox для Сочи
        start_date="2023-06-01"
    )

    # Моделирование на 30 дней
    print("Начало моделирования...")
    model.simulate_period(30)

    # Визуализация результатов
    print("\nВизуализация результатов:")
    model.plot_comparison()

    # Интерактивная карта
    print("\nСоздание интерактивной карты...")
    map_obj = model.plot_interactive_map()
    if map_obj:
        display(map_obj)

    # Анимация
    print("\nГенерация анимации...")
    ani = model.plot_erosion_animation()
    if ani:
        display(HTML(ani.to_jshtml()))

    # Статистика
    stats = model.get_erosion_stats()
    print("\nСтатистика изменений:")
    for k, v in stats.items():
        print(f"{k:>20}: {v:.2f}")