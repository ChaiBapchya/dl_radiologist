import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

SM = nn.Softmax(dim=-1)

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""  
	with torch.no_grad():
		incorrect = 0.0
		batch_size = target.size(0)
		# print(output, target)
		sm_output = SM(output)
		argmaxes=sm_output.argmax(dim=1)
		op_all_good, target_all_good = sm_output[argmaxes==0], target[argmaxes==0]
    
		if target_all_good.size()[0] > 0:
		  incorrect += 14.0 * (1-target_all_good[:,0]).sum()
		# print(target_all_good, correct)

		op_problem, target_problem = sm_output[argmaxes!=0], target[argmaxes!=0]
		if op_problem.size()[0] > 0:
		  op_problem[:,0] = 0
		  # print(op_problem)
		  problem_max = op_problem.max(dim=1)[0]/1.5 #tunable parameter
		  problem_max = problem_max.view(-1,1)
		  # print(problem_max)
		  op_problem[op_problem > problem_max] = 1.0
		  op_problem[op_problem <= problem_max] = 0.0

		  wrong_count = (op_problem-target_problem).abs()
		  wrong_count[wrong_count < 0.6] = 0
		  incorrect += wrong_count.sum()

		correct = 1.0 - (incorrect.item()/(target.size(0)*target.size(1)))

		return correct*100.0


def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10, batch_times=None, data_times=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	batch_times = []
	data_times = []
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		# print(type(target))
		# print(len(target))
		# print(target.size())
		# measure data loading time
		data_time.update(time.time() - end)
		data_times.append(data_time.val)
		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		print("This is while Training")
		output1, output2 = model(input)
		print(output1.size())
		print(output2.size())
		output1 = output1.double()
		output2 = output2.double()
		loss1 = criterion(output1, target)
		loss2 = criterion(output2, target)
		loss = loss1 + loss2
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		batch_times.append(batch_time.val)
		end = time.time()


		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output1, target), target.size(0))

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, batch_times, data_times


def evaluate(model, device, data_loader, criterion, print_freq=10, batch_times=None):
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			print("This is while Evaluating")
			output = model(input)
			output = output.double()
			print(output.size())
			# print(output2.size())

			loss1 = criterion(output, target)
			loss = loss1

			# measure elapsed time
			batch_time.update(time.time() - end)
			batch_times.append(batch_time.val)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target), target.size(0))

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))

			if i % print_freq == 0 or i == len(data_loader)-1:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, results, batch_times
