import os, glob, inspect, time, math, torch

import numpy as np
import matplotlib.pyplot as plt
import source.loss_functions as lfs

from torch.nn import functional as F
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names, savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(15, 10))

    for i in range(num_cont):
        plt.subplot(2,3,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(latent, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def histogram(contents, savename=""):

    n1, _, _ = plt.hist(contents[0], bins=100, alpha=0.5, label='Normal')
    n2, _, _ = plt.hist(contents[1], bins=100, alpha=0.5, label='Abnormal')
    h_inter = np.sum(np.minimum(n1, n2)) / np.sum(n1)
    plt.xlabel("MSE")
    plt.ylabel("Number of Data")
    xmax = max(contents[0].max(), contents[1].max())
    plt.xlim(0, xmax)
    plt.text(x=xmax*0.01, y=max(n1.max(), n2.max()), s="Histogram Intersection: %.3f" %(h_inter))
    plt.legend(loc='upper right')
    plt.savefig(savename)
    plt.close()

def save_graph(contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def torch2npy(input):

    input = input.cpu()
    output = input.detach().numpy()
    return output

def forward(neuralnet, x, z):

    z_code, z_mu, z_sigma = neuralnet.encoder(x.to(neuralnet.device))
    x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
    x_fake = neuralnet.decoder(z.to(neuralnet.device))

    x_mulin, x_mulout = None, None
    for idx_h, hypotheses in enumerate(neuralnet.hypotheses):
        if(idx_h == 0):
            x_fake = hypotheses(x_fake.to(neuralnet.device))
        else:
            tmp_hypo = hypotheses(x_hat.to(neuralnet.device))

            if(x_mulin is None):
                x_mulin = x.unsqueeze(0)
                x_mulout = tmp_hypo.unsqueeze(0)
            else:
                x_mulin = torch.cat((x_mulin, x.unsqueeze(0)))
                x_mulout = torch.cat((x_mulout, tmp_hypo.unsqueeze(0)))
    best_idx = lfs.find_best_x(x_mulin, x_mulout)
    x_best = x_mulout[best_idx]

    x_others = None
    for idx_h, _ in enumerate(x_mulout):
        if(idx_h == best_idx): pass
        else:
            tmp_npy = torch2npy(x_mulout[idx_h])
            if(x_others is None): x_others = x_mulout[idx_h].unsqueeze(0)
            else: x_others = torch.cat((x_others, x_mulout[idx_h].unsqueeze(0)))

    d_real, d_fake, d_best, d_others = 0, 0, 0, 0
    d_real = neuralnet.discriminator(x.to(neuralnet.device))
    d_fake = neuralnet.discriminator(x_fake.to(neuralnet.device))
    d_best = neuralnet.discriminator(x_best.to(neuralnet.device))

    for idx_h, _ in enumerate(x_others):
        tmp_d = neuralnet.discriminator(x_others[idx_h].to(neuralnet.device))
        d_others = torch.add(tmp_d, d_others)
    d_others = torch.div(d_others, neuralnet.num_h-1)

    return d_real, d_fake, d_best, d_others, x_best, x_fake, x_mulout, z_code, z_mu, z_sigma

def training(neuralnet, dataset, epochs, batch_size):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    make_dir(path="results")
    result_list = ["tr_latent", "tr_resotring", "comparing"]
    for result_name in result_list: make_dir(path=os.path.join("results", result_name))

    start_time = time.time()

    iteration = 0
    writer = SummaryWriter()

    test_sq = 20
    test_size = test_sq**2
    list_d, list_g, list_tot = [], [], []
    for epoch in range(epochs):

        x_tr, x_tr_torch, y_tr, y_tr_torch, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        z_tr, z_tr_torch = dataset.random_noise(test_size, neuralnet.z_dim)

        d_real, d_fake, d_best, d_others, x_best, x_fake, x_mulout, z_code, z_mu, z_sigma = \
            forward(neuralnet, x_tr_torch, z_tr_torch)

        z_code = torch2npy(z_code)
        x_best = np.transpose(torch2npy(x_best), (0, 2, 3, 1))
        x_fake = np.transpose(torch2npy(x_fake), (0, 2, 3, 1))
        x_hypo0 = np.transpose(torch2npy(x_mulout[0]), (0, 2, 3, 1))
        x_hypo1 = np.transpose(torch2npy(x_mulout[1]), (0, 2, 3, 1))
        x_hypo2 = np.transpose(torch2npy(x_mulout[2]), (0, 2, 3, 1))

        if(neuralnet.z_dim == 2):
            latent_plot(latent=z_code, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent", "%08d.png" %(epoch)))
        else:
            pca = PCA(n_components=2)
            try:
                pca_features = pca.fit_transform(z_code)
                latent_plot(latent=pca_features, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent", "%08d.png" %(epoch)))
            except: pass

        save_img(contents=[x_tr, x_best, (x_tr-x_best)**2, x_tr, x_fake, (x_tr-x_fake)**2], \
            names=["Real X", "Best of Restorated X", "Difference", "Real X", "Fake X (from Z)", "Difference"], \
            savename=os.path.join("results", "tr_resotring", "%08d.png" %(epoch)))
        save_img(contents=[x_tr, x_best, x_fake, x_hypo0, x_hypo1, x_hypo2], \
            names=["Real X", "Best", "Fake", "0", "1", "2"], \
            savename=os.path.join("results", "comparing", "%08d.png" %(epoch)))

        if(neuralnet.z_dim == 2):
            x_values = np.linspace(-3, 3, test_sq)
            y_values = np.linspace(-3, 3, test_sq)
            z_latents = None
            for y_loc, y_val in enumerate(y_values):
                for x_loc, x_val in enumerate(x_values):
                    z_latent = np.reshape(np.array([y_val, x_val], dtype=np.float32), (1, neuralnet.z_dim))
                    if(z_latents is None): z_latents = z_latent
                    else: z_latents = np.append(z_latents, z_latent, axis=0)
            x_samples = neuralnet.decoder(torch.from_numpy(z_latents).to(neuralnet.device))
            x_samples = np.transpose(torch2npy(x_samples), (0, 2, 3, 1))
            plt.imsave(os.path.join("results", "tr_latent_walk", "%08d.png" %(epoch)), dat2canvas(data=x_samples))

        while(True):
            x_tr, x_tr_torch, y_tr, y_tr_torch, terminator = dataset.next_train(batch_size)
            z_tr, z_tr_torch = dataset.random_noise(batch_size, neuralnet.z_dim)

            d_real, d_fake, d_best, d_others, x_best, x_fake, x_mulout, z_code, z_mu, z_sigma = \
                forward(neuralnet, x_tr_torch, z_tr_torch)
            loss_d = lfs.lossfunc_d(d_real, d_fake, d_best, d_others, neuralnet.num_h)
            neuralnet.optimizer_d.zero_grad()
            loss_d.backward(retain_graph=True)
            neuralnet.optimizer_d.step()

            d_real, d_fake, d_best, d_others, x_best, x_fake, x_mulout, z_code, z_mu, z_sigma = \
                forward(neuralnet, x_tr_torch, z_tr_torch)
            loss_g = lfs.lossfunc_g(x_tr_torch, x_best, z_mu, z_sigma, loss_d)
            neuralnet.optimizer_g.zero_grad()
            loss_g.backward()
            neuralnet.optimizer_g.step()

            z_code = torch2npy(z_code)
            x_best = np.transpose(torch2npy(x_best), (0, 2, 3, 1))
            x_fake = np.transpose(torch2npy(x_fake), (0, 2, 3, 1))

            loss_tot = loss_d+loss_g
            list_d.append(loss_d)
            list_g.append(loss_g)
            list_tot.append(loss_tot)

            writer.add_scalar('ConAD/loss_d', loss_d, iteration)
            writer.add_scalar('ConAD/loss_g', loss_g, iteration)
            writer.add_scalar('ConAD/loss_total', loss_tot, iteration)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  D:%.3f, G:%.3f, Total:%.3f" \
            %(epoch, epochs, iteration, loss_d, loss_g, loss_tot))
        for idx_m, model in enumerate(neuralnet.models):
            torch.save(model.state_dict(), PACK_PATH+"/runs/params-%d" %(idx_m))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    save_graph(contents=list_d, xlabel="Iteration", ylabel="Discriminate Error", savename="loss_d")
    save_graph(contents=list_g, xlabel="Iteration", ylabel="Generate Error", savename="loss_g")
    save_graph(contents=list_tot, xlabel="Iteration", ylabel="Total Loss", savename="loss_tot")

def test(neuralnet, dataset):

    param_paths = glob.glob(os.path.join(PACK_PATH, "runs", "params*"))
    param_paths.sort()

    if(len(param_paths) > 0):
        for idx_p, param_path in enumerate(param_paths):
            print(PACK_PATH+"/runs/params-%d" %(idx_p))
            neuralnet.models[idx_p].load_state_dict(torch.load(PACK_PATH+"/runs/params-%d" %(idx_p)))
            neuralnet.models[idx_p].eval()

    print("\nTest...")

    make_dir(path="test")
    result_list = ["inbound", "outbound"]
    for result_name in result_list: make_dir(path=os.path.join("test", result_name))

    scores_normal, scores_abnormal = [], []
    while(True):
        x_te, x_te_torch, y_te, y_te_torch, terminator = dataset.next_test(1) # y_te does not used in this prj.
        z_te, z_te_torch = dataset.random_noise(1, neuralnet.z_dim)

        d_real, d_fake, d_best, d_others, x_best, x_fake, x_mulout, z_code, z_mu, z_sigma = \
            forward(neuralnet, x_te_torch, z_te_torch)
        mse = lfs.mean_square_error(x_te_torch, x_best)

        score_anomaly = mse.item()

        if(y_te == 1): scores_normal.append(score_anomaly)
        else: scores_abnormal.append(score_anomaly)

        if(terminator): break

    scores_normal = np.asarray(scores_normal)
    scores_abnormal = np.asarray(scores_abnormal)
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    abnormal_avg, abnormal_std = np.average(scores_abnormal), np.std(scores_abnormal)
    print("Noraml  avg: %.5f, std: %.5f" %(normal_avg, normal_std))
    print("Abnoraml  avg: %.5f, std: %.5f" %(abnormal_avg, abnormal_std))
    outbound = normal_avg + (normal_std * 3)
    print("Outlier boundary of normal data: %.5f" %(outbound))

    histogram(contents=[scores_normal, scores_abnormal], savename="histogram-test.png")

    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    z_code_tot, y_te_tot = None, None
    loss4box = [[], [], [], [], [], [], [], [], [], []]
    while(True):
        x_te, x_te_torch, y_te, y_te_torch, terminator = dataset.next_test(1) # y_te does not used in this prj.
        z_te, z_te_torch = dataset.random_noise(1, neuralnet.z_dim)

        d_real, d_fake, d_best, d_others, x_best, x_fake, x_mulout, z_code, z_mu, z_sigma = \
            forward(neuralnet, x_te_torch, z_te_torch)
        mse = lfs.mean_square_error(x_te_torch, x_best)

        score_anomaly = mse.item()

        z_code = torch2npy(z_code)
        x_best = np.transpose(torch2npy(x_best), (0, 2, 3, 1))
        x_fake = np.transpose(torch2npy(x_fake), (0, 2, 3, 1))

        loss4box[y_te[0]].append(score_anomaly)

        if(z_code_tot is None):
            z_code_tot = z_code
            y_te_tot = y_te
        else:
            z_code_tot = np.append(z_code_tot, z_code, axis=0)
            y_te_tot = np.append(y_te_tot, y_te, axis=0)

        outcheck = score_anomaly > outbound
        fcsv.write("%d, %.3f, %r\n" %(y_te, score_anomaly, outcheck))

        [h, w, c] = x_te[0].shape
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = x_best[0]
        canvas[:, w*2:, :] = (x_te[0]-x_best[0])**2
        if(outcheck):
            plt.imsave(os.path.join("test", "outbound", "%08d-%08d.png" %(testnum, int(score_anomaly))), gray2rgb(gray=canvas))
        else:
            plt.imsave(os.path.join("test", "inbound", "%08d-%08d.png" %(testnum, int(score_anomaly))), gray2rgb(gray=canvas))

        testnum += 1

        if(terminator): break

    boxplot(contents=loss4box, savename="test-box.png")

    if(neuralnet.z_dim == 2):
        latent_plot(latent=z_code_tot, y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join("test-latent.png"))
    else:
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(z_code_tot)
        latent_plot(latent=pca_features, y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join("test-latent.png"))
