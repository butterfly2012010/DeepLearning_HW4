"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import argparse
import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    get_loaders_seg,
    plot_couple_examples,
    calculate_top1_accuracy_seg,
)
from loss import YoloLoss
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='obj', type=str, help='training task')
parser.add_argument('--epochs', default=5, type=int, help='set number of training epochs')
args = parser.parse_args()


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch):
    loop = tqdm(train_loader, leave=True)
    losses = []
    # accuracy = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out, out_seg = model(x)
            
            ### debug
            # print(out)
            
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
            
        losses.append(loss.item())
        # accuracy.append(mapval.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        # mean_accuracy = sum(accuracy) / len(accuracy)
        loop.set_postfix(loss=mean_loss)
        # loop.set_postfix(acc=mean_accuracy)
        
    with open('./train_loss_obj.txt', mode='a') as f:
            f.write(f"{epoch} {mean_loss:.4f}\n")
                

def train_seg(train_loader, model, optimizer, loss_fn, scaler, epoch):
    loop = tqdm(train_loader, leave=True)
    losses = []
    accuracy = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)  # imgs
        y = y.to(config.DEVICE)  # masks
        # y0, y1, y2 = (
        #     y[0].to(config.DEVICE),
        #     y[1].to(config.DEVICE),
        #     y[2].to(config.DEVICE),
        # )

        with torch.cuda.amp.autocast():
            obj_out, seg_out = model(x)
            
            ### debug
            # print(seg_out)
            # print(y.size())
            # print(seg_out.size())
            
            # pred = A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE)(seg_out)
            pred = F.interpolate(seg_out, size=(config.IMAGE_SIZE, config.IMAGE_SIZE), mode='bilinear')
            loss = loss_fn(pred, y.argmax(dim=1))
            acc = calculate_top1_accuracy_seg(true_labels=y, predicted_labels=pred)

        losses.append(loss.item())
        accuracy.append(acc)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        mean_accuracy = sum(accuracy) / len(accuracy)
        loop.set_postfix(loss=mean_loss)
        loop.set_postfix(acc=mean_accuracy)
        
    with open('./train_loss_seg.txt', mode='a') as f:
            f.write(f"{epoch} {mean_loss:.4f}\n")
    with open('./train_acc_seg.txt', mode='a') as f:
            f.write(f"{epoch} {mean_accuracy:.4f}\n")


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )
    
    train_loader_ade20k, test_loader_ade20k, train_eval_loader_ade20k = get_loaders_seg(
        train_csv_path=config.SEG_DATASET + "/train.txt", test_csv_path=config.SEG_DATASET + "/test.txt"
    )

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
    #     )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(args.epochs):  # config.NUM_EPOCHS
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        # if epoch % 2 != 0:
        #     # 需要加上freeze weight的功能
        #     train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        # else:
        #     # 需要加上freeze weight的功能
        #     train_seg(train_loader_ade20k, model, optimizer, loss_fn = torch.nn.CrossEntropyLoss(), scaler = scaler)
            
        if args.task == 'obj':
            # 需要加上freeze weight的功能
            train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch=epoch)
        else:
            # 需要加上freeze weight的功能
            train_seg(train_loader_ade20k, model, optimizer, loss_fn=torch.nn.CrossEntropyLoss(), scaler=scaler, epoch=epoch)
        
        if epoch % 5 == 0:  # epoch % 10
            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer, filename=f"checkpoint_final.pth")


            
            
        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #check_class_accuracy(model, train_eval_loader, threshold=config.CONF_THRESHOLD)
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if args.task == 'obj' and epoch % 1 == 0:# and epoch > 0:  # epoch % 5
            # print("On Test loader:")
            acc_test = check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)
            with open('./train_acc_obj.txt', mode='a') as f:
                f.write(f"{epoch} {acc_test.item():.4f}\n")
            acc_test = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            with open('./test_acc_obj.txt', mode='a') as f:
                f.write(f"{epoch} {acc_test.item():.4f}\n")

            # pred_boxes, true_boxes = get_evaluation_bboxes(
            #     test_loader,
            #     model,
            #     iou_threshold=config.NMS_IOU_THRESH,
            #     anchors=config.ANCHORS,
            #     threshold=config.CONF_THRESHOLD,
            # )
            # mapval = mean_average_precision(
            #     pred_boxes,
            #     true_boxes,
            #     iou_threshold=config.MAP_IOU_THRESH,
            #     box_format="midpoint",
            #     num_classes=config.NUM_CLASSES,
            # )
            # print(f"MAP: {mapval.item()}")
            

            ### BATCH_SIZE = 16, CUDA OOM
            # torch.cuda.empty_cache()
            
            


if __name__ == "__main__":
    main()
