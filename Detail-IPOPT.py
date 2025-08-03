import numpy as np
import matplotlib.pyplot as plt

###پکیج ها برای زبان فارسی
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager
# تنظیمات فونت برای نمایش صحیح فارسی در نمودارها
# تنظیم فونت فارسی
font_path = 'C:\\Windows\\Fonts\\Tahoma.ttf'  # مسیر فونت
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False # برای نمایش صحیح علامت منفی


def prepare_persian_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# --- بخش ۱: تعریف توابع مسئله (بدون تغییر) ---
def objective_function(x):
    return -x[0] - x[1]

def objective_gradient(x):
    return np.array([-1.0, -1.0])

def objective_hessian(x):
    return np.zeros((2, 2))

def constraints(x):
    g1 = -x[0]**3 + 6*x[0]**2 - 9*x[0] + x[1] - 10
    g2 = -x[1] + 14
    g3 = x[0] - 5
    return np.array([g1, g2, g3])

def constraints_jacobian(x):
    jac_g1 = np.array([-3*x[0]**2 + 12*x[0] - 9, 1.0])
    jac_g2 = np.array([0.0, -1.0])
    jac_g3 = np.array([1.0, 0.0])
    return np.array([jac_g1, jac_g2, jac_g3])

def constraints_hessian(x):
    hess_g1 = np.array([[-6*x[0] + 12, 0.0], [0.0, 0.0]])
    hess_g2 = np.zeros((2, 2))
    hess_g3 = np.zeros((2, 2))
    return np.array([hess_g1, hess_g2, hess_g3])


# --- بخش ۲: پیاده‌سازی الگوریتم با خروجی دقیق ---
def barrier_method_solver_detailed(x0, mu0=10.0, sigma=0.1, tolerance=1e-5, max_outer_iter=15, max_inner_iter=20):
    x = np.array(x0, dtype=float)
    mu = mu0
    path = [x.copy()]

    print("="*60)
    print(prepare_persian_text("--- شروع الگوریتم نقطه داخلی با خروجی مرحله به مرحله ---"))
    print(prepare_persian_text(f"نقطه شروع: x = [{x[0]:.4f}, {x[1]:.4f}], پارامتر مانع اولیه: μ = {mu:.2f}"))
    print("="*60 + "\n")

    for i in range(max_outer_iter):
        print(prepare_persian_text(f"--- تکرار اصلی {i+1}: کاهش پارامتر مانع ---"))
        print(prepare_persian_text(f"مقدار جدید پارامتر مانع: μ = {mu:.6f}"))
        print(prepare_persian_text("   حل زیرمسئله با روش نیوتن:"))
        print("   Iter\t   x_k\t\t\t f(x)\t\t||grad_P||\t alpha")
        print("   -----------------------------------------------------------------------")

        for j in range(max_inner_iter):
            g = constraints(x)
            if np.any(g >= 0):
                print(prepare_persian_text(f"خطا در تکرار {j+1}: نقطه خارج از ناحیه ممکن است. g(x) = {g}"))
                return x, path
            
            grad_barrier = -np.sum(constraints_jacobian(x) / g[:, np.newaxis], axis=0)
            hess_barrier_1 = np.sum(
                (constraints_jacobian(x)[:, :, np.newaxis] * constraints_jacobian(x)[:, np.newaxis, :]) / (g**2)[:, np.newaxis, np.newaxis],
                axis=0
            )
            hess_barrier_2 = -np.sum(constraints_hessian(x) / g[:, np.newaxis, np.newaxis], axis=0)
            
            hess_barrier = hess_barrier_1 + hess_barrier_2
            
            total_grad = objective_gradient(x) - mu * grad_barrier
            total_hess = objective_hessian(x) - mu * hess_barrier

            grad_norm = np.linalg.norm(total_grad)
            
            try:
                step = np.linalg.solve(total_hess, -total_grad)
            except np.linalg.LinAlgError:
                print(prepare_persian_text("خطا: ماتریس هسین منفرد است."))
                return x, path

            alpha = 1.0
            beta = 0.5
            while np.any(constraints(x + alpha * step) >= -1e-9):
                alpha *= beta
                if alpha < 1e-8:
                    break
            
            print(f"   {j+1}\t [{x[0]:.4f}, {x[1]:.4f}]\t {objective_function(x):.4f}\t{grad_norm:.2e}\t {alpha:.3f}")
            
            x = x + alpha * step
            path.append(x.copy())

            if grad_norm < tolerance:
                print(prepare_persian_text(f"   همگرایی برای μ={mu:.6f} در نقطه x=[{x[0]:.4f}, {x[1]:.4f}] حاصل شد.\n"))
                break
        
        if mu < tolerance:
            print("\n" + "="*60)
            print(prepare_persian_text("--- پایان الگوریتم: همگرایی نهایی حاصل شد ---"))
            print("="*60)
            return x, path

        mu *= sigma

    print(prepare_persian_text("حداکثر تکرارها انجام شد."))
    return x, path

# --- بخش ۳: اجرا و مصورسازی ---
# نقطه شروع (باید کاملاً در داخل ناحیه ممکن باشد، یعنی تمام قیود اکیداً کوچکتر از صفر باشند)
initial_point = [4.5, 18.0] 

optimal_x, solution_path = barrier_method_solver_detailed(initial_point)
solution_path = np.array(solution_path)

print(prepare_persian_text(f"\nنقطه بهینه یافت‌شده: x1 = {optimal_x[0]:.6f}, x2 = {optimal_x[1]:.6f}"))
print(prepare_persian_text(f"مقدار بهینه تابع هدف: {objective_function(optimal_x):.6f}"))

# ... (رسم نمودار ) ...
plt.rcParams['font.family'] = 'Tahoma'
fig, ax = plt.subplots(figsize=(12, 10))
x1_vals = np.linspace(-1, 6, 400)
x2_vals = np.linspace(13, 35, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
G1 = -X1**3 + 6*X1**2 - 9*X1 + X2 - 10
G2 = -X2 + 14
G3 = X1 - 5
feasible_region = (G1 <= 0) & (G2 <= 0) & (G3 <= 0)
ax.contourf(X1, X2, feasible_region, levels=[0.5, 1.5], colors=['lightblue'], alpha=0.5)
Z = -X1 - X2
contours = ax.contour(X1, X2, Z, levels=20, colors='gray', alpha=0.7)
ax.clabel(contours, inline=True, fontsize=9)
ax.plot(solution_path[:, 0], solution_path[:, 1], 'o-', color='black', label=prepare_persian_text('مسیر الگوریتم نقطه داخلی'), markersize=4, lw=2)
ax.plot(solution_path[0, 0], solution_path[0, 1], 'go', markersize=12, label=prepare_persian_text('نقطه شروع'))
ax.plot(solution_path[-1, 0], solution_path[-1, 1], 'r*', markersize=15, label=prepare_persian_text('نقطه بهینه یافت‌شده'))
ax.set_title(prepare_persian_text('شبیه‌سازی گام به گام روش نقطه داخلی (Barrier Method)'))
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)
ax.set_xlim(-1, 6)
ax.set_ylim(13, 35)
plt.savefig("barrier_method_path_detailed.png")
print(prepare_persian_text("\nنمودار مسیر حل در فایل 'barrier_method_path_detailed.png' ذخیره شد."))
plt.show()