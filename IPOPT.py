#پکیج های مصورسازی
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#پکیج pyomo
from pyomo.environ import *
###پکیج ها برای زبان فارسی
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager

# تنظیمات فونت برای نمایش صحیح فارسی در نمودارها
# تنظیم فونت فارسی
font_path = 'C:\\Windows\\Fonts\\Tahoma.ttf'  # مسیر فونت
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False # برای نمایش صحیح علامت منفی

# =============================================================================
# بخش ۱: حل مسئله با Pyomo
# =============================================================================

# 1. ساخت یک مدل
model = ConcreteModel()

# 2. تعریف متغیرها با نقطه شروع ممکن (feasible)
# within=Reals: دامنه متغیرها را اعداد حقیقی تعریف می‌کند.
# initialize: یک نقطه شروع اولیه برای الگوریتم حل تعیین می‌کند
initial_point = {'x1': 4.0, 'x2': 14.0}
model.x1 = Var(within=Reals, initialize=initial_point['x1'])
model.x2 = Var(within=Reals, initialize=initial_point['x2'])

# 3. تعریف تابع هدف
model.objective = Objective(expr = -model.x1 - model.x2, sense=minimize)

# 4. تعریف قیود
model.constraint1 = Constraint(expr = -model.x1**3 + 6*model.x1**2 - 9*model.x1 + model.x2 - 10 <= 0)
model.constraint2 = Constraint(expr = -model.x2 + 14 <= 0)
model.constraint3 = Constraint(expr = model.x1 - 5 <= 0)

# 5. انتخاب و استفاده از حل‌کننده
solver = SolverFactory('ipopt')

# متغیر برای ذخیره نقطه بهینه
optimal_point = None

try:
    results = solver.solve(model, tee=True) # tee=True تا خروجی حل‌کننده چاپ شود
    model.solutions.load_from(results)

    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print("راه حل بهینه با موفقیت پیدا شد.")
        final_x1 = value(model.x1)
        final_x2 = value(model.x2)
        optimal_point = {'x1': final_x1, 'x2': final_x2}
        print(f"نقطه بهینه: x1 = {final_x1:.4f}, x2 = {final_x2:.4f}")
        print(f"مقدار بهینه تابع هدف: {value(model.objective):.4f}")
    else:
        print("حل‌کننده نتوانست به جواب بهینه برسد.")
        print("شرط پایان:", results.solver.termination_condition)

except Exception as e:
    print(f"یک خطا در هنگام اجرای حل‌کننده رخ داد: {e}")

# =============================================================================
# بخش ۲: مصورسازی و ایجاد انیمیشن (فقط در صورت موفقیت‌آمیز بودن حل)
# =============================================================================

if optimal_point:
    print("\nدر حال ساخت انیمیشن...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # --- رسم پس‌زمینه نمودار (ناحیه ممکن و خطوط تراز) ---
    x1_vals = np.linspace(-1, 6, 400)
    x2_vals = np.linspace(10, 20, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    # محاسبه قیود برای نمایش ناحیه ممکن
    G1 = -X1**3 + 6*X1**2 - 9*X1 + X2 - 10
    G2 = -X2 + 14
    G3 = X1 - 5
    feasible_region = (G1 <= 0) & (G2 <= 0) & (G3 <= 0)
    ax.contourf(X1, X2, feasible_region, levels=[0.5, 1.5], colors=['lightblue'], alpha=0.5)
    # تابع تبدیل متن فارسی
    def prepare_persian_text(text):
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
        
    # محاسبه و رسم خطوط تراز تابع هدف
    Z = -X1 - X2
    contours = ax.contour(X1, X2, Z, levels=20, colors='gray', alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=8)

    # --- تنظیمات انیمیشن ---
    start_pos = np.array([initial_point['x1'], initial_point['x2']])
    end_pos = np.array([optimal_point['x1'], optimal_point['x2']])

    # رسم نقاط شروع و پایان
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label=prepare_persian_text('نقطه شروع'))
    ax.plot(end_pos[0], end_pos[1], 'ro', markersize=10, label=prepare_persian_text('نقطه بهینه'))
    
    # تعریف نقطه متحرک برای انیمیشن
    moving_point, = ax.plot([], [], 'o-', color='black', lw=2, markersize=8)

    # تابع انیمیشن که در هر فریم فراخوانی می‌شود
    def animate(i):
        # محاسبه موقعیت نقطه در طول مسیر مستقیم
        num_frames = 100
        current_pos = start_pos + (end_pos - start_pos) * (i / num_frames)
        # رسم مسیر از شروع تا نقطه فعلی
        path_x = [start_pos[0], current_pos[0]]
        path_y = [start_pos[1], current_pos[1]]
        moving_point.set_data(path_x, path_y)
        return moving_point,

    # ایجاد و ذخیره انیمیشن
    anim = FuncAnimation(fig, animate, frames=101, interval=40, blit=True)
    ax.set_title(prepare_persian_text('شبیه‌سازی مسیر بهینه‌سازی (Pyomo + IPOPT)'))

    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    try:
        anim_file = 'pyomo_solution_path.gif'
        anim.save(anim_file, writer='pillow')
        print(f"\nانیمیشن با موفقیت در فایل '{anim_file}' ذخیره شد.")
        plt.show()
    except Exception as e:
        print(f"\nخطا در ذخیره انیمیشن: {e}")
        print("ممکن است نیاز به نصب کتابخانه 'pillow' داشته باشید: pip install pillow")