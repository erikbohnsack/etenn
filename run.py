import sys
import os

if os.path.exists("/Users/erikbohnsack/Code/MOT/conventional-MOT/"):
    sys.path.append("/Users/erikbohnsack/Code/MOT/conventional-MOT/")
else:
    sys.path.append('/home/mlt/mot/conventional-MOT')
from eval_post_fafe_pp import eval_post_fafe_pp
from eval_post_fafe import eval_post_fafe
import platform
from pathlib import Path
import datetime
from data_utils import kitti_stuff
from utils.plot_stuff import plot_tracking_history
import pickle
from utils import logger
from utils import plot_stuff

def run(names_to_run, sequences, plot_tracks, plot_sequence_analysis):
    if platform.system() == 'Darwin':
        fafe_model_path = '/Users/erikbohnsack/Code/MOT/fafe/trained_models/weights_2019-05-03_14-01_epoch_110_fafe'
        pp_model_path = '/Users/erikbohnsack/Code/MOT/fafe/trained_models/weights_2019-05-03_14-01_epoch_110_pp'
        data_path = '/Users/erikbohnsack/data'
        config_path = '/Users/erikbohnsack/Code/MOT/fafe/trained_models/config_2019-05-03_14-01.yml'
    else:
        fafe_model_paths = {
            'bev_NN': '/home/mlt/mot/fafe/trained_models/2019-05-07_09:40_bev_fafe/weights_2019-05-07_09-40_epoch_299',
            'pp_NN': '/home/mlt/mot/fafe/trained_models/2019-05-07_11:11_pp_fafe/weights_2019-05-07_11-11_epoch_299_fafe',
            'bev_nn': '/home/mlt/mot/fafe/trained_models/2019-05-19_10:04_bev_little/weights_2019-05-19_10-04_epoch_95',
            'pp_nn': '/home/mlt/mot/fafe/trained_models/2019-05-06_13:05_pp_little/weights_2019-05-06_13-05_epoch_299_fafe'
        }
        pp_model_paths = {
            'bev_NN': '/home/mlt/mot/fafe/trained_models/2019-05-07_09:40_bev_fafe/weights_2019-05-07_09-40_epoch_299',
            'pp_NN': '/home/mlt/mot/fafe/trained_models/2019-05-07_11:11_pp_fafe/weights_2019-05-07_11-11_epoch_299_pp',
            'bev_nn': '/home/mlt/mot/fafe/trained_models/2019-05-19_10:04_bev_little/weights_2019-05-19_10-04_epoch_95',
            'pp_nn': '/home/mlt/mot/fafe/trained_models/2019-05-06_13:05_pp_little/weights_2019-05-06_13-05_epoch_299_pp'
        }
        config_paths = {
            'bev_NN': '/home/mlt/mot/fafe/trained_models/2019-05-07_09:40_bev_fafe/config_2019-05-07_09-40.yml',
            'pp_NN': '/home/mlt/mot/fafe/trained_models/2019-05-07_11:11_pp_fafe/config_2019-05-07_11-11.yml',
            'bev_nn': '/home/mlt/mot/fafe/trained_models/2019-05-19_10:04_bev_little/config_2019-05-19_10-04.yml',
            'pp_nn': '/home/mlt/mot/fafe/trained_models/2019-05-06_13:05_pp_little/config_2019-05-06_13-05.yml'}
        data_path = '/home/mlt/data'


    pps = {'bev_NN' : False,
           'pp_NN' : True,
           'bev_nn' : False,
           'pp_nn' : True}


    ##########################################
    # HARD CODED
    num_conseq_frames = 5
    ##########################################

    for idx, name in enumerate(names):
        print('Now running: {}'.format(name))
        pp = pps[name]
        fafe_model_path = fafe_model_paths[name]
        pp_model_path = pp_model_paths[name]
        config_path = config_paths[name]
        filename = Path(config_path).stem
        time_str = datetime.datetime.now().strftime('%m-%d_%H%M')
        showroom_path = os.path.join('showroom', filename + '_' + time_str)
        if not os.path.exists(showroom_path):
            os.mkdir(showroom_path)
        logpath = os.path.join(showroom_path, 'logs')
        if not os.path.exists(logpath):
            os.mkdir(logpath)
        output_tracks_dir = os.path.join(showroom_path, 'output_tracks')
        if not os.path.exists(output_tracks_dir):
            os.mkdir(output_tracks_dir)

        kitti = kitti_stuff.Kitti(ROOT=data_path, split='training')
        for sequence in sequences:
            ##############################
            # Load data
            #############################
            kitti.imus = kitti.load_imu(sequence)
            kitti.lbls = kitti.load_labels(sequence)

            ##############################
            # Inference stuff
            #############################
            if pp:
                data, total_time_per_iteration = eval_post_fafe_pp(fafe_model_path,
                                                                   pp_model_path,
                                                                   data_path,
                                                                   config_path,
                                                                   sequence,
                                                                   kitti)
            else:
                data, total_time_per_iteration = eval_post_fafe(fafe_model_path,
                                                                data_path,
                                                                config_path,
                                                                sequence,
                                                                kitti)

            datapath = os.path.join(logpath, 'log-seq' + str(sequence).zfill(4))
            with open(datapath, 'ab') as fp:
                pickle.dump(data, fp)

            ##############################
            # Stats stuff
            #############################
            gospa_sl = logger.calculate_GOSPA_score(data=data, gt_dims=2)
            mot_summary = logger.calculate_MOT(sequence, data_path, data=data, classes_to_track=['Car', 'Van'])
            pred_gospa_scores, pred_mean_gospas = logger.fafe_prediction_stats(sequence, kitti, data=data,
                                                                               num_conseq_frames=num_conseq_frames)

            stats = {'config_name': name,
                     'sequence_idx': sequence,
                     'time_per_iter': total_time_per_iteration,
                     'gospa_sl': gospa_sl,
                     'mot_summary': mot_summary,
                     'motion_model': 'N/A',
                     'poisson_states_model_name': 'N/A',
                     'filter_name': 'N/A',
                     'predictions_average_gospa': pred_mean_gospas}
            statspath = os.path.join(logpath, 'stats_fafe_seq_' + str(sequence).zfill(4))
            with open(statspath, 'wb') as fp:
                pickle.dump(stats, fp)

            ##############################
            # Plot Tracks
            #############################
            if plot_tracks:
                plot_path = os.path.join(output_tracks_dir, str(sequence).zfill(4))
                print('Saving tracking history plots...')
                for dict_item in data:
                    frame = dict_item['current_time']
                    print('{},'.format(frame), end='')
                    plot_tracking_history(plot_path, sequence_idx=sequence, num_conseq_frames=5, data=data, kitti=kitti,
                                          final_frame_idx=frame, disp='save', only_alive=True, show_cov=False,
                                          show_predictions=True,
                                          fafe=True, car_van_flag=True)

        ###################################
        # Sequence analysis for one config
        ###################################
        if plot_sequence_analysis:
            df, avg_df = plot_stuff.sequence_analysis(filenames_prefix=logpath + '/stats', sortby='CfgName')
            if avg_df is not None: print(avg_df.to_string())
            df

            # Save as latex tables
            df2 = df.drop(
                columns=['Filter', 'PoissonModel', 'MotionModel', 'MostlyLost', '#Fragmentations', 'PredGOSPA'])
            file = open(os.path.join(showroom_path, "stats_latex_table.txt"), "w")

            for seq in range(0, max(df['SeqId'].values) + 1):
                df3 = df2.loc[df2['SeqId'] == seq]
                df3 = df3.sort_values(by='CfgName', ascending=True)
                _str = '\n\\begin{table}[] \n\centering'
                file.write(_str)
                file.write(df3.to_latex(index=False))
                _str = ' \caption{Results for sequence ' + str(seq) + '} \n \label{tab:avg-pmbm-df}\n\end{table}\n'
                file.write(_str)
            file.close()

            if avg_df is not None:
                file = open(os.path.join(showroom_path, "/average_stats_latex_table.txt"), "w")
                file.write(avg_df.to_latex(index=True))
                file.close()

if __name__ == "__main__":
    names = ['bev_NN', 'bev_nn', 'pp_NN', 'pp_nn']
    sequences = [0]
    plot_tracks = True
    plot_sequence_analysis = True

    run(names_to_run=names,
        sequences=sequences,
        plot_tracks=plot_tracks,
        plot_sequence_analysis=plot_sequence_analysis)


















