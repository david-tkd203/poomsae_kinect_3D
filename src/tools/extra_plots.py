"""src/tools/extra_plots.py

Generador de gráficos exploratorios a partir de `resumen_metricas.csv`.

Genera (por defecto) los siguientes gráficos en la carpeta de salida:
 - hist_seg_score_mean.png (histograma de seg_score_mean)
 - boxplot_seg_score_mean.png (boxplot de seg_score_mean)
 - scatter_score_vs_duration.png (scatter seg_score_mean vs seg_duration_mean, color por n_segments)
 - corr_heatmap.png (mapa de correlación entre variables numéricas)
 - pca_2d.png (PCA 2D sobre variables numéricas)
 - top10_by_score.png (barra top10 muestras por seg_score_mean)
 - hist_n_segments.png (histograma de n_segments)

Uso:

    python -m src.tools.extra_plots --resumen reports/analisis/resumen_metricas.csv --out-dir reports/analisis/extra_plots

"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas.plotting import scatter_matrix
import math

def _print_header(title: str) -> None:
    print('\n' + '='*60)
    print(title)
    print('-'*60)


def ensure_out(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def load_resumen(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def hist_seg_score_mean(df: pd.DataFrame, outdir: Path) -> None:
    s = df['seg_score_mean'].dropna()
    _print_header('Histograma de seg_score_mean')
    print(f'count={len(s)}, mean={s.mean():.4f}, median={s.median():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}')
    plt.figure(figsize=(6, 4))
    plt.hist(s, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('seg_score_mean')
    plt.ylabel('Count')
    plt.title('Histograma de seg_score_mean')
    plt.tight_layout()
    plt.savefig(outdir / 'hist_seg_score_mean.png', dpi=200)
    plt.close()


def boxplot_seg_score_mean(df: pd.DataFrame, outdir: Path) -> None:
    s = df['seg_score_mean'].dropna()
    _print_header('Boxplot de seg_score_mean')
    q = s.quantile([0.25, 0.5, 0.75])
    print(f'Q1={q.loc[0.25]:.4f}, Q2(median)={q.loc[0.5]:.4f}, Q3={q.loc[0.75]:.4f}')
    plt.figure(figsize=(6, 4))
    plt.boxplot(s, vert=False)
    plt.xlabel('seg_score_mean')
    plt.title('Boxplot de seg_score_mean')
    plt.tight_layout()
    plt.savefig(outdir / 'boxplot_seg_score_mean.png', dpi=200)
    plt.close()


def scatter_score_vs_duration(df: pd.DataFrame, outdir: Path) -> None:
    x = df['seg_duration_mean']
    y = df['seg_score_mean']
    s = df['n_segments'].fillna(df['n_segments'].median())
    # normalize sizes
    sizes = (s - s.min()) / max(1.0, (s.max() - s.min())) * 200 + 20

    # Regression fit for analysis
    mask = (~x.isna()) & (~y.isna())
    xm = x[mask].astype(float)
    ym = y[mask].astype(float)
    _print_header('Scatter seg_score_mean vs seg_duration_mean')
    if len(xm) >= 2:
        coef = np.polyfit(xm, ym, 1)
        slope, intercept = coef[0], coef[1]
        pred = slope * xm + intercept
        ss_res = ((ym - pred) ** 2).sum()
        ss_tot = ((ym - ym.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        print(f'n={len(xm)}, slope={slope:.6f}, intercept={intercept:.6f}, R2={r2:.4f}')
    else:
        print('No hay suficientes puntos para regresión.')

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(x, y, s=sizes, c=s, cmap='viridis', alpha=0.8, edgecolor='k')
    if len(xm) >= 2:
        xs = np.linspace(np.nanmin(xm), np.nanmax(xm), 100)
        ys = slope * xs + intercept
        plt.plot(xs, ys, color='red', linewidth=1.5, label=f'fit R2={r2:.3f}')
        plt.legend()
    plt.colorbar(sc, label='n_segments')
    plt.xlabel('seg_duration_mean [s]')
    plt.ylabel('seg_score_mean')
    plt.title('Score vs Duración media de segmento (tamaño=color n_segments)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'scatter_score_vs_duration.png', dpi=200)
    plt.close()


def corr_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=90)
    plt.yticks(ticks, corr.columns)
    plt.title('Correlation heatmap (variables numéricas)')
    plt.tight_layout()
    plt.savefig(outdir / 'corr_heatmap.png', dpi=200)
    plt.close()
    # Print top correlated pairs
    _print_header('Top correlations (abs value)')
    flat = []
    for i, a in enumerate(corr.columns):
        for j, b in enumerate(corr.columns):
            if j <= i:
                continue
            flat.append((a, b, corr.iloc[i, j]))
    flat_sorted = sorted(flat, key=lambda x: abs(x[2]), reverse=True)
    for a, b, v in flat_sorted[:10]:
        print(f'{a} <-> {b}: corr={v:.4f}')


def pca_2d(df: pd.DataFrame, outdir: Path) -> None:
    num = df.select_dtypes(include=[np.number]).copy()
    # drop columns with NaNs entirely or fill
    num = num.fillna(num.mean())
    if num.shape[1] < 2:
        return
    # center
    X = num.values.astype(float)
    Xc = X - X.mean(axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :2] * S[:2]

    # explained variance
    var_ratio = (S**2) / np.sum(S**2)
    _print_header('PCA 2D')
    print(f'Explained variance (PC1,PC2) = {var_ratio[0]:.4f}, {var_ratio[1]:.4f}')
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=df.index, cmap='tab20', alpha=0.8)
    for i, txt in enumerate(df.get('sample_id', df.index.astype(str))):
        if i % max(1, len(df)//20) == 0:
            plt.text(coords[i, 0], coords[i, 1], str(txt), fontsize=7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA 2D (num. features)')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(outdir / 'pca_2d.png', dpi=200)
    plt.close()


def top10_by_score(df: pd.DataFrame, outdir: Path) -> None:
    df2 = df[['sample_id', 'seg_score_mean']].dropna().sort_values('seg_score_mean', ascending=False)
    top = df2.head(10)
    _print_header('Top 10 muestras por seg_score_mean')
    print(top.to_string(index=False))
    plt.figure(figsize=(8, 4))
    plt.barh(top['sample_id'].astype(str), top['seg_score_mean'], color='seagreen')
    plt.gca().invert_yaxis()
    plt.xlabel('seg_score_mean')
    plt.title('Top 10 muestras por seg_score_mean')
    plt.tight_layout()
    plt.savefig(outdir / 'top10_by_score.png', dpi=200)
    plt.close()


def hist_n_segments(df: pd.DataFrame, outdir: Path) -> None:
    s = df['n_segments'].dropna()
    _print_header('Histograma de n_segments')
    print(f'count={len(s)}, mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}, min={s.min()}, max={s.max()}')
    plt.figure(figsize=(6, 4))
    plt.hist(s, bins=20, color='orchid', edgecolor='black')
    plt.xlabel('n_segments')
    plt.ylabel('Count')
    plt.title('Histograma de n_segments')
    plt.tight_layout()
    plt.savefig(outdir / 'hist_n_segments.png', dpi=200)
    plt.close()


def pairwise_scatter_matrix(df: pd.DataFrame, outdir: Path) -> None:
    num = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).mean())
    cols = num.columns[:6] if num.shape[1] > 6 else num.columns
    _print_header('Scatter matrix - mostrando correlaciones principales')
    corr = num[cols].corr()
    print(corr)
    try:
        plt.figure(figsize=(10, 10))
        # use histogram on diagonal to avoid KDE failures on singular data
        scatter_matrix(num[cols], alpha=0.6, diagonal='hist', figsize=(10, 10))
        plt.suptitle('Scatter matrix (primeras columnas numéricas)')
        plt.tight_layout()
        plt.savefig(outdir / 'scatter_matrix.png', dpi=200)
        plt.close()
    except Exception as e:
        print('Warning: scatter_matrix failed:', e)
        # fallback: pairwise scatter plots for first 4 columns
        try:
            small = num[cols[:4]] if len(cols) >= 4 else num[cols]
            plt.figure(figsize=(8, 8))
            scatter_matrix(small, alpha=0.6, diagonal='hist', figsize=(8, 8))
            plt.suptitle('Scatter matrix (fallback)')
            plt.tight_layout()
            plt.savefig(outdir / 'scatter_matrix_fallback.png', dpi=200)
            plt.close()
        except Exception as e2:
            print('Fallback scatter matrix also failed:', e2)


def kde_and_moments(df: pd.DataFrame, outdir: Path) -> None:
    s = df['seg_score_mean'].dropna()
    _print_header('KDE y momentos de seg_score_mean')
    print(f'skew={s.skew():.4f}, kurtosis={s.kurtosis():.4f}')
    plt.figure(figsize=(6, 4))
    s.plot.kde()
    plt.xlabel('seg_score_mean')
    plt.title('KDE de seg_score_mean')
    plt.tight_layout()
    plt.savefig(outdir / 'kde_seg_score_mean.png', dpi=200)
    plt.close()


def generate_all(resumen_csv: Path, out_dir: Path) -> None:
    out = Path(out_dir)
    ensure_out(out)
    df = load_resumen(Path(resumen_csv))

    # Plots
    hist_seg_score_mean(df, out)
    boxplot_seg_score_mean(df, out)
    scatter_score_vs_duration(df, out)
    corr_heatmap(df, out)
    pca_2d(df, out)
    top10_by_score(df, out)
    hist_n_segments(df, out)
    pairwise_scatter_matrix(df, out)
    kde_and_moments(df, out)


def build_argparser():
    ap = argparse.ArgumentParser(description='Generador de gráficos exploratorios desde resumen_metricas.csv')
    ap.add_argument('--resumen', required=True, help='CSV resumen (reports/analisis/resumen_metricas.csv)')
    ap.add_argument('--out-dir', required=False, default='reports/analisis/extra_plots', help='Directorio salida PNGs')
    return ap


def main(argv=None):
    ap = build_argparser()
    args = ap.parse_args(argv)
    resumen = Path(args.resumen)
    out = Path(args.out_dir)
    if not resumen.exists():
        raise SystemExit(f"No existe resumen CSV: {resumen}")
    generate_all(resumen, out)
    print(f"Gráficos generados en: {out}")


if __name__ == '__main__':
    main()
