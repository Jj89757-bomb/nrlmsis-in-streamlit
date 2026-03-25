import streamlit as st
from pymsis import msis
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 物理常数
BOLTZMANN = 1.380649e-23  # J/K


# 模型输出参数索引
# 0: 总质量密度 (kg/m^3)
# 1: N2 数密度 (m^-3)
# 2: O2 数密度 (m^-3)
# 3: O  数密度 (m^-3)
# 4: He 数密度 (m^-3)
# 5: H  数密度 (m^-3)
# 6: Ar 数密度 (m^-3)
# 7: N  数密度 (m^-3)
# 8: 异常氧数密度 (m^-3)
# 9: NO 数密度 (m^-3)
# 10: 温度 (K)

def calculate_pressure(nd, temp):
    """根据总分子数密度和温度计算压强 (Pa)"""
    return nd * BOLTZMANN * temp


def compute_profile(dt, lon, lat, alt_range, alt_step,
                    f107s=None, f107as=None, aps=None):
    """
    计算指定时间、经纬度下，高度变化的大气参数剖面
    返回: 高度数组 (km), 温度 (K), 总质量密度 (kg/m^3), 压强 (Pa)
    """
    alts = np.arange(alt_range[0], alt_range[1] + alt_step, alt_step)
    data = msis.run(dt, lon, lat, alts,
                    f107s=f107s, f107as=f107as, aps=aps)
    data = data.squeeze()  # 形状 (n_alts, 11)

    temp = data[:, 10]
    mass_density = data[:, 0]
    number_densities = data[:, 1:10]
    total_nd = np.sum(number_densities, axis=1)
    pressure = calculate_pressure(total_nd, temp)

    return alts, temp, mass_density, pressure


def compute_global_grid(dt, alt, lon_range, lon_step, lat_range, lat_step, param_idx,
                        f107s=None, f107as=None, aps=None):
    """
    计算指定时间、高度下，全球网格上的某一参数分布
    param_idx: 0 -> 总质量密度, 10 -> 温度, 或 'pressure' 计算压强
    返回: 经度数组, 纬度数组, 二维参数数组
    """
    lons = np.arange(lon_range[0], lon_range[1] + lon_step, lon_step)
    lats = np.arange(lat_range[0], lat_range[1] + lat_step, lat_step)
    data = msis.run(dt, lons, lats, alt,
                    f107s=f107s, f107as=f107as, aps=aps)
    data = data.squeeze()  # 形状 (n_lons, n_lats, 11)

    if param_idx == 'pressure':
        number_densities = data[:, :, 1:10]
        total_nd = np.sum(number_densities, axis=2)
        temp = data[:, :, 10]
        param = calculate_pressure(total_nd, temp)
    else:
        param = data[:, :, param_idx]

    return lons, lats, param


def main():
    st.set_page_config(page_title="NRLMSIS 大气模型演示", layout="wide")
    st.title("🌍 NRLMSIS 大气模型交互界面")
    st.markdown("本应用基于 NRLMSIS 模型，计算指定时间、位置的大气参数，并可视化随高度变化曲线及全球分布。")

    # ==================== 侧边栏：用户输入 ====================
    st.sidebar.header("输入参数")

    # 时间选择
    date_input = st.sidebar.date_input("日期", datetime.date.today())
    time_input = st.sidebar.time_input("时间 (UTC)", datetime.time(12, 0))
    user_datetime = datetime.datetime.combine(date_input, time_input)
    st.sidebar.write(f"选定时刻: {user_datetime.strftime('%Y-%m-%d %H:%M UTC')}")

    # 剖面图设置
    st.sidebar.subheader("剖面图设置 (固定位置，高度变化)")
    lon_profile = st.sidebar.number_input("经度 (°)", value=0.0, step=1.0, format="%.1f")
    lat_profile = st.sidebar.number_input("纬度 (°)", value=45.0, step=1.0, format="%.1f")
    alt_min = st.sidebar.number_input("最小高度 (km)", value=0.0, step=10.0)
    alt_max = st.sidebar.number_input("最大高度 (km)", value=1000.0, step=50.0)
    alt_step = st.sidebar.number_input("高度步长 (km)", value=10.0, step=5.0)
    if alt_max <= alt_min:
        st.sidebar.error("最大高度必须大于最小高度")
        st.stop()

    # 全球图设置
    st.sidebar.subheader("全球分布设置 (固定时间、高度)")
    alt_global = st.sidebar.number_input("高度 (km) [用于全球图]", value=200.0, step=10.0)
    lon_min = st.sidebar.number_input("经度范围起点 (°)", value=-180.0)
    lon_max = st.sidebar.number_input("经度范围终点 (°)", value=180.0)
    lon_step_global = st.sidebar.number_input("经度步长 (°)", value=10.0, min_value=1.0, step=1.0)
    lat_min = st.sidebar.number_input("纬度范围起点 (°)", value=-90.0)
    lat_max = st.sidebar.number_input("纬度范围终点 (°)", value=90.0)
    lat_step_global = st.sidebar.number_input("纬度步长 (°)", value=10.0, min_value=1.0, step=1.0)

    # 太阳活动指数（可选）
    st.sidebar.subheader("太阳/地磁活动指数 (可选，留空则使用模型自动获取的历史数据)")
    use_custom_indices = st.sidebar.checkbox("自定义指数")
    f107s = None
    f107as = None
    aps = None
    if use_custom_indices:
        st.sidebar.markdown("**F10.7 指数**")
        f107s_val = st.sidebar.number_input("每日 F10.7 (前一天值)", value=150.0, step=10.0)
        f107as_val = st.sidebar.number_input("81天平均 F10.7 (中心日期)", value=150.0, step=10.0)
        # 构造形状 (1,) 的数组
        f107s = np.array([f107s_val])
        f107as = np.array([f107as_val])

        st.sidebar.markdown("**Ap 指数** (需要7个值，对应于每日Ap及6个3小时ap指数)")
        ap_input_method = st.sidebar.radio("Ap 输入方式", ["单个值（自动复制7次）", "手动输入7个值（逗号分隔）"])
        if ap_input_method == "单个值（自动复制7次）":
            ap_daily = st.sidebar.number_input("每日 Ap 值", value=4.0, step=1.0)
            # 构造形状 (1, 7) 的数组，所有7个值相同
            aps = np.full((1, 7), ap_daily, dtype=float)
        else:
            ap_string = st.sidebar.text_input("输入7个Ap值（逗号分隔）", "4,4,4,4,4,4,4")
            try:
                ap_vals = [float(x.strip()) for x in ap_string.split(',')]
                if len(ap_vals) != 7:
                    st.sidebar.error("必须输入恰好7个数值")
                else:
                    aps = np.array([ap_vals], dtype=float)  # 形状 (1, 7)
            except:
                st.sidebar.error("输入格式错误，请使用逗号分隔的7个数字")

    # ==================== 主要展示区域 ====================
    tab1, tab2 = st.tabs(["📈 大气参数剖面", "🌐 全球分布图"])

    with tab1:
        st.subheader(
            f"大气参数随高度变化 (时间: {user_datetime.strftime('%Y-%m-%d %H:%M')}, 位置: ({lon_profile}°, {lat_profile}°))")
        with st.spinner("正在计算剖面数据..."):
            try:
                alts, temp, mass_density, pressure = compute_profile(
                    user_datetime, lon_profile, lat_profile,
                    (alt_min, alt_max), alt_step,
                    f107s=f107s, f107as=f107as, aps=aps
                )
                # 绘图
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].plot(temp, alts, 'r-', linewidth=2)
                axes[0].set_xlabel('温度 (K)')
                axes[0].set_ylabel('高度 (km)')
                axes[0].grid(True, linestyle='--', alpha=0.7)
                axes[0].set_title('温度剖面')

                axes[1].plot(mass_density, alts, 'b-', linewidth=2)
                axes[1].set_xlabel('总质量密度 (kg/m³)')
                axes[1].set_ylabel('高度 (km)')
                axes[1].set_xscale('log')
                axes[1].grid(True, linestyle='--', alpha=0.7)
                axes[1].set_title('密度剖面')

                axes[2].plot(pressure, alts, 'g-', linewidth=2)
                axes[2].set_xlabel('压强 (Pa)')
                axes[2].set_ylabel('高度 (km)')
                axes[2].set_xscale('log')
                axes[2].grid(True, linestyle='--', alpha=0.7)
                axes[2].set_title('压强剖面')

                plt.tight_layout()
                st.pyplot(fig)

                with st.expander("查看详细数据"):
                    df = pd.DataFrame({
                        '高度 (km)': alts,
                        '温度 (K)': temp,
                        '总质量密度 (kg/m³)': mass_density,
                        '压强 (Pa)': pressure
                    })
                    st.dataframe(df)
            except Exception as e:
                st.error(f"计算剖面时出错: {e}")

    with tab2:
        st.subheader(f"大气参数全球分布 (时间: {user_datetime.strftime('%Y-%m-%d %H:%M')}, 高度: {alt_global} km)")

        param_options = {
            "总质量密度 (kg/m³)": 0,
            "N₂数密度 (m⁻³)": 1,
            "O₂数密度 (m⁻³)": 2,
            "O数密度 (m⁻³)": 3,
            "He数密度 (m⁻³)": 4,
            "H数密度 (m⁻³)": 5,
            "Ar数密度 (m⁻³)": 6,
            "N数密度 (m⁻³)": 7,
            "异常氧数密度 (m⁻³)": 8,
            "NO数密度 (m⁻³)": 9,
            "温度 (K)": 10,
            "压强 (Pa)": "pressure",
        }
        selected_param = st.selectbox("选择参数", list(param_options.keys()))
        param_idx = param_options[selected_param]

        with st.spinner("正在计算全球数据，可能需要几秒钟..."):
            try:
                lons, lats, param_grid = compute_global_grid(
                    user_datetime, alt_global,
                    (lon_min, lon_max), lon_step_global,
                    (lat_min, lat_max), lat_step_global,
                    param_idx,
                    f107s=f107s, f107as=f107as, aps=aps
                )

                st.write(f"经度点数: {len(lons)}, 纬度点数: {len(lats)}")
                st.write(f"参数网格形状: {param_grid.shape}")

                fig, ax = plt.subplots(figsize=(10, 6))
                if selected_param in ["总质量密度 (kg/m³)", "压强 (Pa)"]:
                    valid_data = param_grid[param_grid > 0]
                    vmin = valid_data.min() if len(valid_data) > 0 else 1e-10
                    norm = colors.LogNorm(vmin=vmin, vmax=param_grid.max())
                else:
                    norm = None

                im = ax.imshow(param_grid.T,
                               extent=[lons[0], lons[-1], lats[0], lats[-1]],
                               origin='lower',
                               aspect='auto',
                               cmap='viridis',
                               norm=norm)
                ax.set_xlabel('经度 (°)')
                ax.set_ylabel('纬度 (°)')
                ax.set_title(f'{selected_param} 全球分布 @ {alt_global} km')
                ax.grid(True, linestyle='--', alpha=0.5)
                plt.colorbar(im, ax=ax, label=selected_param)

                # 若安装了 cartopy 则绘制带海岸线版本
                try:
                    import cartopy.crs as ccrs
                    fig_map = plt.figure(figsize=(10, 6))
                    ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                    ax_map.imshow(param_grid.T,
                                  extent=[lons[0], lons[-1], lats[0], lats[-1]],
                                  origin='lower',
                                  transform=ccrs.PlateCarree(),
                                  cmap='viridis',
                                  norm=norm)
                    ax_map.coastlines()
                    ax_map.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
                    plt.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.05, label=selected_param)
                    ax_map.set_title(f'{selected_param} 全球分布 (cartopy) @ {alt_global} km')
                    st.pyplot(fig_map)
                except ImportError:
                    st.pyplot(fig)

                st.info(f"数据范围: {param_grid.min():.3e} ~ {param_grid.max():.3e}")
                csv_data = pd.DataFrame(param_grid, index=lons, columns=lats)
                csv_data.index.name = '经度'
                csv_data.columns.name = '纬度'
                st.download_button(
                    label="下载全球数据 (CSV)",
                    data=csv_data.to_csv().encode('utf-8'),
                    file_name=f"nrlmsis_global_{selected_param.replace(' ', '_')}_{user_datetime.strftime('%Y%m%d_%H%M')}_alt{alt_global}km.csv",
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"计算全球分布时出错: {e}")


if __name__ == "__main__":
    main()