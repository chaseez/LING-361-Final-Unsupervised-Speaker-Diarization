from pyannote.metrics.detection import DetectionErrorRate, DetectionPrecision, DetectionRecall, DetectionPrecisionRecallFMeasure
from pyannote.database.loader import load_rttm
from pyannote.core import Annotation, Segment
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch


def create_annotations(activity_tensor: torch.Tensor, total_frames: int, frames_per_window: int, window_duration: float) -> list[Annotation]:
    """
    Convert speaker activity matrix to pyannote Annotation timeline
    
    Args:
        activity_tensor: (num_speakers, num_frames) binary activation matrix
        total_frames: Original audio duration in frames
        frames_per_window: Processing window size used in model
        window_duration: Real-time duration of each window in seconds
    
    Returns:
        Annotation: Pyannote segmentation object with speaker intervals
    """
    # Calculate time per frame (constant across all windows)
    frame_duration = window_duration / frames_per_window  # ~0.15789 seconds
    
    # Initialize list to hold Annotation objects
    annotations = Annotation()
    
    # Process each speaker's activity pattern
    for speaker_idx, activity in enumerate(activity_tensor):
        start_frame = None
        
        # Convert tensor to numpy array for easier handling
        activity = activity.numpy()
        
        for frame_idx, active in enumerate(activity):
            if active == 1:
                if start_frame is None:  # Start new segment
                    start_frame = frame_idx
            else:
                if start_frame is not None:  # End current segment
                    # Calculate absolute times in seconds
                    start_time = start_frame * frame_duration
                    end_time = frame_idx * frame_duration
                    annotations[Segment(start_time, end_time)] = f"spk{speaker_idx:02d}"
                    start_frame = None
                    
        # Handle final segment if active at end
        if start_frame is not None:
            start_time = start_frame * frame_duration
            end_time = total_frames * frame_duration
            annotations[Segment(start_time, end_time)] = f"spk{speaker_idx:02d}"
        
        # annotations.append(annotation)
    
    return annotations


if __name__ == '__main__':
    for path in Path('models').iterdir():
        words = path.as_posix().split('-')
        # Rename all the files to remove the loss values
        if words[-1].isnumeric():
            words.remove(words[-1])
            words[-1] = words[-1] + '.pt'
            new_path = '-'.join(words)
            # print(new_path)
            path.rename(new_path)

    # Get all the starting files
    model_path = list(Path('models').rglob('*-0-*-100-epochs*.pt'))

    results = []
    for model in tqdm(model_path):
        # load in the activation matrix generated from training
        activation = torch.load(model, map_location='cpu')
        activation.requires_grad_(False)

        for i in range(1,1000):
            words = model.as_posix().split('-')
            words[1] = str(i)
            # get the next activation matrix (next window for that audio segment)
            new_model = '-'.join(words)
            try:
                activation = torch.cat([activation,torch.load(new_model, map_location='cpu')], dim=-1)
            except Exception as e:
                break

        # print(activation.shape)
        name = model.name.split('-')[0]
        annotation = load_rttm(f"../data/voxconverse-master/dev/{name}.rttm")[name]
        num_speakers = len(annotation.labels())

        relevant_activation = activation[:num_speakers].contiguous()
        pred_annotation = (relevant_activation != 0).long()
        pred_annotation = create_annotations(pred_annotation, pred_annotation.shape[1], 38, 6)
        
        # get the diarization error rate within 0.5 seconds of the start and end of each segment
        metric = DetectionErrorRate(collar=0.5, skip_overlap=False)
        der = metric(annotation, pred_annotation)

        # get the precision within 0.5 seconds of the start and end of each segment
        metric = DetectionPrecision(collar=0.5, skip_overlap=False)
        precision = metric(annotation, pred_annotation)

        # get the recall rate within 0.5 seconds of the start and end of each segment
        metric = DetectionRecall(collar=0.5, skip_overlap=False)
        recall = metric(annotation, pred_annotation)

        # get the F1 score within 0.5 seconds of the start and end of each segment
        metric = DetectionPrecisionRecallFMeasure(collar=0.5, skip_overlap=False)
        F1 = metric(annotation, pred_annotation)
        
        results.append((der, precision, recall, F1))
    
    res_string = 'Mean ({:.3f}) Min ({:.3f}) Median ({:.3f}) Max ({:.3f}) Stdev ({:.3f}) Var ({:.3f})'

    der, precision, recall, F1 = zip(*results)
    print('DER:', res_string.format(np.mean(der), np.min(der), np.median(der), np.max(der), np.std(der), np.var(der)), sep='\t')
    print('Precision:', res_string.format(np.mean(precision), np.min(precision), np.median(precision), np.max(precision), np.std(precision), np.var(precision)))
    print('Recall:', res_string.format(np.mean(recall), np.min(recall), np.median(recall), np.max(recall), np.std(recall), np.var(recall)), sep='\t')
    print('F1:', res_string.format(np.mean(F1), np.min(F1), np.median(F1), np.max(F1), np.std(F1), np.var(F1)), sep='\t')