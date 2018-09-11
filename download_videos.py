from pytube import YouTube
import os
import glob

def download_video(link, save_dir):
    print('Download %s to %s ...' %(link, save_dir))
    YouTube(link).streams.first().download(save_dir)
    print('Done\n')


links0 = ['https://www.youtube.com/watch?v=NasyGUeNMTs',
          'https://www.youtube.com/watch?v=pU3iGpwKxKc',
          'https://www.youtube.com/watch?v=6TdtxElNCtI',
          'https://www.youtube.com/watch?v=lL74n-Vr91k',
          'https://www.youtube.com/watch?v=oAuOjG4L1Ng',] # Kizuna AI

links1 = ['https://www.youtube.com/watch?v=0V1vk83iV-o',
          'https://www.youtube.com/watch?v=TwMkoEuQNAk',
          'https://www.youtube.com/watch?v=b_SEEnVq_GM',
          'https://www.youtube.com/watch?v=ce7Xy8wvMzI',
          'https://www.youtube.com/watch?v=2L7X1UQFWgI'] # Mirai Akari

links2 = ['https://www.youtube.com/watch?v=TeKTVFgw1hM',
          'https://www.youtube.com/watch?v=dzEk6wZ4Xuc',
          'https://www.youtube.com/watch?v=zdneuijW_70',
          'https://www.youtube.com/watch?v=GG7nBgIHmKw',
          'https://www.youtube.com/watch?v=ZJinxt-wui0'] # Kaguya Luna

links3 = ['https://www.youtube.com/watch?v=fLC5TE_KYcw',
          'https://www.youtube.com/watch?v=KmfGNTbMNBk',
          'https://www.youtube.com/watch?v=t1V8O7q0bA8',
          'https://www.youtube.com/watch?v=lqUQWwK3Xag',
          'https://www.youtube.com/watch?v=vcxW5AcyAWU'] # Siro

links4 = ['https://www.youtube.com/watch?v=cqncAh_28Es',
          'https://www.youtube.com/watch?v=DoVh4Fc43Bo',
          'https://www.youtube.com/watch?v=0q4CQEw60IM',
          'https://www.youtube.com/watch?v=L5sy3wgNwaI',
          'https://www.youtube.com/watch?v=QDWKOzum6F8'] # Neko Mas

links_all = [links0, links1, links2, links3, links4]

save_dirs = ['./video/KizunaAI',
             './video/MiraiAkari',
             './video/KaguyaLuna',
             './video/Siro',
             './video/NekoMas']

for (links, dir) in zip(links_all, save_dirs):
    for link in links:
        download_video(link, dir)

    videos = glob.glob(os.path.join(dir, '*.mp4'))
    for (n, video) in enumerate(videos):
        os.rename(video, os.path.join(dir, 'video-{:02}.mp4'.format(n)))
