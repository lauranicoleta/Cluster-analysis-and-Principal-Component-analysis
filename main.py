from sklearn.decomposition import PCA
from utils import *

def execute():
    # incarc setul de date din fișierul CSV
    tabel = pd.read_csv('C:\\Users\\popla\\OneDrive\\Desktop\\Facultate anul III\\Semestrul I\\Dezvoltare software pentru analiza datelor\\POP_LAURA_1094\\Pop_Laura_1094_1\\dataIN\\winequality-red-white.csv', sep=";", index_col=0)

    # obtin lista de instanțe și variabile
    instante = tabel.index
    variabile = tabel.columns[1:]  # exclud prima coloană cu indexul

    n = len(instante)
    m = len(variabile)

    # inlocuiesc valorile lipsă (NaN) cu mediana fiecărei variabile
    nan_replace(tabel)

    # trnasform dataframe-ul pandas intr-un numpy array
    x = tabel[variabile].values

    # construiesc ierarhie
    h = hclust.linkage(x, method='ward') #metoda ward este o metoda de minimizare a variatiei totala in clusteri, ceea ce duce la formarea de clusteri omogeni si compacti

    # gasesc numărul optim de clusteri
    p = n - 1
    k_diff_max = np.argmax(h[1:, 2] - h[:(p - 1), 2])
    nr_clusteri = p - k_diff_max

    # obtin partitia optima si o salvez intr-un fișier CSV
    partitie_optima = partitie(h, nr_clusteri, p, instante)
    partitie_optima_t = pd.DataFrame(data={"Cluster": partitie_optima}, index=instante)
    partitie_optima_t.to_csv("PartitieOptima.csv")

    # obtin și salvez partitia cu 3 clusteri
    partitie_3 = partitie(h, 3, p, instante)
    partitie_3_t = pd.DataFrame(data={"Cluster": partitie_3}, index=instante)
    partitie_3_t.to_csv("Partitie3.csv")

    # obtin și salvez partitia cu 4 clusteri
    partitie_4 = partitie(h, 4, p, instante)
    partitie_4_t = pd.DataFrame(data={"Cluster": partitie_4}, index=instante)
    partitie_4_t.to_csv("Partitie4.csv")

    # desenez histograma pentru primele 3 variabile in functie de partitia optimă
    for i in range(3):
        histograma(x[:, i], variabile[i], partitie_optima)

    # afisez toate desenele
    show()

    t = pd.read_csv("C:\\Users\\popla\\OneDrive\\Desktop\\Facultate anul III\\Semestrul I\\Dezvoltare software pentru analiza datelor\\POP_LAURA_1094\\Pop_Laura_1094_1\\dataIN\\letter-recognition.csv", sep=",", index_col=0)
    nan_replace(t)

    variable_observate = t.columns[0:]

    # standardizare set de date
    x = (t[variable_observate] - np.mean(t[variable_observate], axis=0)) / np.std(t[variable_observate], axis=0)
    n, m = x.shape

    model_acp = PCA()
    model_acp.fit(x)

    # x = setul de date observate
    # alpha = valorile proprii
    # a = vectori proprii
    # c = componente principale rezultate in urma ACP: c = x @ a
    alpha = model_acp.explained_variance_
    a = model_acp.components_
    c = model_acp.transform(x)
    print("alpha", alpha)
    print("a", a)
    print("componente principale", c)

    etichete = ["C" + str(i + 1) for i in range(len(alpha))]
    componente_tabelar = tabelare_matrice(c, t.index, etichete, "componente.csv")
    plot_componente(componente_tabelar, "C1", "C2")

    # criterii de identificare a numarului de componente semnificative
    # Kaiser
    where = np.where(alpha > 1)
    # print(where)
    nr_comp_kaiser = len(where[0])
    print("Numar componente principale semnificative conform crit. Kaiser:", nr_comp_kaiser)

    # procent de acoperire
    ponderi = np.cumsum(alpha / sum(alpha))
    where = np.where(ponderi > 0.8)
    nr_comp_procent = where[0][0] + 1
    print("Numar componente principale semnificative conform crit. Procente acoperire:", nr_comp_procent)

    # Cattell
    eps = alpha[:(m - 1)] - alpha[1:]
    sigma = eps[:(m - 2)] - eps[1:]
    negative = sigma < 0

    if any(negative):
        where = np.where(negative)
        nr_comp_cattell = where[0][0] + 1
    else:
        nr_comp_cattell = 0
    print("Numar componente principale semnificative conform crit. Cattell:", nr_comp_cattell)

    # calcul corelatii intre variabilele observate si componentele principale
    corr = np.corrcoef(x, c, rowvar=False)
    print(corr)
    print(x.shape, corr.shape)
    r_x_c = corr[:m, :m]
    r_x_c_tabelar = tabelare_matrice(r_x_c, variable_observate, etichete,
                                     "corelatii_factoriale.csv")
    corelograma(r_x_c_tabelar)
    plot_corelatii(r_x_c_tabelar, "C1", "C2")
    plot_corelatii(r_x_c_tabelar, "C1", "C3")
    plot_corelatii(r_x_c_tabelar, "C3", "C4")
    plot_corelatii(r_x_c_tabelar, "C5", "C6")


    # calcul cosinusuri
    componete_patrat = c * c
    cosin = np.transpose(componete_patrat.T / np.sum(componete_patrat, axis=1))
    cosin_tabelar = tabelare_matrice(cosin, t.index, etichete, "cosin.csv")

    # calcul comunalitati
    r_x_c_patrat = r_x_c * r_x_c
    comunalitati = np.cumsum(r_x_c_patrat, axis=1)
    comunalitati_tabelar = tabelare_matrice(comunalitati, variable_observate, etichete,
                                            "comunalitati.csv")

    show()

if __name__ == '__main__':
    execute()
