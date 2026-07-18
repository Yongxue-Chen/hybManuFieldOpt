import torch
import torch.nn as nn
import numpy as np
import pyvista as pv
from itertools import cycle
from siren_pytorch import SirenNet
from torch.utils.data import TensorDataset, DataLoader
import igl

np.bool = np.bool_

class sdfModel:
    def __init__(self, device='cuda', model_load_path = None):
        self.model = SirenNet(3,256,1,5,w0=30,w0_initial=30).to(device)
        if(model_load_path):
            checkPoint = torch.load(model_load_path)
            self.model.load_state_dict(checkPoint['model_state_dict'])
            
        self.device = device
        self.scalarLoss_DistWt = 1e2
        self.scalarLoss_SurfWt = 1e2
        self.classLoss_DistWt = 1e2
        self.dirLoss_SurfWt = 1e0
        self.normLoss_DistWt = 1e0
        self.offLoss_DistWt = 1e1
        self.optimizer = None
        
    def train(self, points_dist, scalars_dist, points_surf, normals_surf, lr = 1e-5, epochNum=60, savePath = "./outSDF.pt"):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        if (not points_dist.requires_grad):
            points_dist.requires_grad = True
            
        if (not points_surf.requires_grad):
            points_surf.requires_grad = True
        
        
        #classification targets: +1=inside (SDF<0), -1=outside (SDF>0)
        classifyVector = torch.ones_like(scalars_dist, device=self.device)  # shape [N]
        outMask = scalars_dist > 0
        classifyVector[outMask] = -1
        classifyVector = classifyVector.unsqueeze(1)  # now shape [N,1], AFTER correct assignment
        
        # Debug: verify distribution
        n_inside  = (~outMask).sum().item()
        n_outside = outMask.sum().item()
        print(f"[Data] inside pts: {n_inside}, outside pts: {n_outside}, ratio: {n_inside/(n_inside+n_outside+1e-8):.3f}")
        

        inputPoints_dist_dataset = TensorDataset(points_dist, scalars_dist, classifyVector)
        inputPoints_dist_dataLoader = DataLoader(inputPoints_dist_dataset, batch_size=3000,shuffle=True)
        
        
        inputPoints_surf_dataset = TensorDataset(points_surf, normals_surf)
        inputPoints_surf_dataLoader = DataLoader(inputPoints_surf_dataset, batch_size=3000,shuffle=True)
        
        if(len(inputPoints_surf_dataLoader)<len(inputPoints_dist_dataLoader)):
            print(len(inputPoints_dist_dataLoader))
            print(len(inputPoints_surf_dataLoader))
            inputPoints_surf_dataLoader = cycle(inputPoints_surf_dataLoader)
        
        for epoch in range(epochNum):
            epochLoss = 0
            scalarLossRecord = 0
            classLossRecord = 0
            eikonalLossRecord = 0
            dirLossRecord = 0
            surfScalarLossRecord = 0
            iterCount = 0
            # if (epoch+1)%50 == 0:
            #     self.dirLoss_SurfWt*=2
                
            # if (epoch+1)%5 == 0:
            #     self.normLoss_DistWt*=2
            #     if(self.normLoss_DistWt>1e-1):
            #         self.normLoss_DistWt = 1e-1
                
            for (batch_points_dist, batch_scalars_dist, batch_class_dist), (batch_points_surf, batch_normals_surf) in zip(inputPoints_dist_dataLoader, inputPoints_surf_dataLoader):
                loss = 0
                
             #############################
             ##Losses on distributed points   
                outScalars = self.model(batch_points_dist)
                outGrads = torch.autograd.grad(outputs=outScalars, inputs=batch_points_dist, grad_outputs=torch.ones_like(outScalars), create_graph=True, retain_graph=True)[0]
            
                #scalar loss
                # scalarError = torch.abs(outScalars - batch_scalars_dist.unsqueeze(1))
                # scalarLoss = torch.mean(scalarError)
                scalarError = outScalars - batch_scalars_dist.unsqueeze(1)
                scalarLoss = torch.mean(scalarError * scalarError)
                loss += self.scalarLoss_DistWt*scalarLoss
                scalarLossRecord += scalarLoss
                
                # classify loss: enforce sign (outside>0, inside<0)
                classError = outScalars*batch_class_dist
                classLoss = torch.mean(torch.relu(classError))
                loss += self.classLoss_DistWt*classLoss
                classLossRecord += classLoss
                
                # DEBUG: print on first batch of first epoch only
                if epoch == 0 and iterCount == 0:
                    print(f"\n[DEBUG classify] batch_class_dist unique: {torch.unique(batch_class_dist).tolist()}")
                    print(f"[DEBUG classify] batch_class_dist: +1 count={( batch_class_dist > 0).sum().item()}, -1 count={(batch_class_dist < 0).sum().item()}")
                    print(f"[DEBUG classify] outScalars range: [{outScalars.min().item():.4f}, {outScalars.max().item():.4f}]")
                    print(f"[DEBUG classify] classError range: [{classError.min().item():.4f}, {classError.max().item():.4f}]")
                    print(f"[DEBUG classify] relu(classError) > 0 count: {(torch.relu(classError) > 0).sum().item()} / {classError.numel()}")
                    print(f"[DEBUG classify] classLoss: {classLoss.item():.6e}")
                    print(f"[DEBUG classify] batch_scalars_dist range: [{batch_scalars_dist.min().item():.4f}, {batch_scalars_dist.max().item():.4f}]")
                    print()
                
                # eikonal loss
                normError = 1 - torch.norm(outGrads, dim=1)
                normLoss = torch.mean(normError*normError)
                loss += self.normLoss_DistWt*normLoss
                eikonalLossRecord += normLoss
            
                
              ###########################
              ##Losses on surface points
                outScalars = self.model(batch_points_surf)
                outGrads = torch.autograd.grad(outputs=outScalars, inputs=batch_points_surf, grad_outputs=torch.ones_like(outScalars), create_graph=True, retain_graph=True)[0]
                
                #gradient dir loss
                dirError = outGrads - batch_normals_surf
                dirLoss = torch.mean(dirError*dirError)
                loss += self.dirLoss_SurfWt*dirLoss
                dirLossRecord += dirLoss

                
                #scalar loss on surface
                scalarError = outScalars
                surfScalarLoss = torch.mean(scalarError*scalarError)
                loss += self.scalarLoss_SurfWt*surfScalarLoss
                surfScalarLossRecord += surfScalarLoss
              
              
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                iterCount+=1
                epochLoss+=loss
                
                if(iterCount%2) == 0:
                    print(".",end="")
            print("\n")
            epochLoss = epochLoss/iterCount
            scalarLossRecord   = scalarLossRecord/iterCount
            classLossRecord    = classLossRecord/iterCount
            eikonalLossRecord  = eikonalLossRecord/iterCount
            dirLossRecord      = dirLossRecord/iterCount
            surfScalarLossRecord = surfScalarLossRecord/iterCount
            if (epoch+1)%1==0:
                print(f"=== Epoch {epoch+1} ===")
                print(f"  total loss       : {epochLoss.item():.6f}")
                print(f"  scalar loss (dist): {scalarLossRecord.item():.3e}  (wt={self.scalarLoss_DistWt:.0e})")
                print(f"  classify loss    : {classLossRecord.item():.3e}  (wt={self.classLoss_DistWt:.0e})")
                print(f"  eikonal loss     : {eikonalLossRecord.item():.3e}  (wt={self.normLoss_DistWt:.0e})")
                print(f"  dir loss (surf)  : {dirLossRecord.item():.3e}  (wt={self.dirLoss_SurfWt:.0e})")
                print(f"  scalar loss (surf): {surfScalarLossRecord.item():.3e}  (wt={self.scalarLoss_SurfWt:.0e})")
            
            if (epoch+1)%5==0:
                writePath = savePath
                checkPoint = {'model_state_dict':self.model.state_dict()}
                torch.save(checkPoint, writePath)
            
        
        
    # def predictOuts(self, points):
    #     outVals = self.model(points)
    #     return outVals
    
    def predictOuts(self, points, batch_size=100):
        outVals_list = []  # To store batch outputs
        num_points = points.shape[0]
    
        for i in range(0, num_points, batch_size):
            batch = points[i:i+batch_size]  # Slice batch
            outScalars = self.model(batch)  # Get output for batch
            outVals_list.append(outScalars)  # Store result
    
        returnVals = torch.cat(outVals_list, dim=0)  # Concatenate along first dimension
        returnVal = {'scalars': returnVals}
        
        return returnVal
    
    def predictGrads(self, points, batch_size=100):

        if not points.requires_grad:
            points.requires_grad_(True)
        outScalars = self.model(points)  # Get output for batch
        outGrads = torch.autograd.grad(outputs=outScalars, inputs=points, grad_outputs=torch.ones_like(outScalars), create_graph=False, retain_graph=False)[0]
        returnVal = {'grads': outGrads}
        
        return returnVal
    
    



def samplePointsNearSurf(mesh,device='cuda'):
    mesh.compute_normals(cell_normals=False, inplace=True)
    boundaryPoints = mesh.points 
    boundaryNormals = mesh.point_data['Normals']
    
    boundaryPoints = torch.tensor(boundaryPoints, dtype=torch.float32)
    boundaryNormals = torch.tensor(boundaryNormals, dtype=torch.float32)
    boundaryNormals = boundaryNormals/torch.norm(boundaryNormals, dim=1).unsqueeze(1)
    
    newDistances = 0.02*torch.rand(boundaryPoints.shape[0])
    newPoints = boundaryPoints + newDistances.unsqueeze(1)*boundaryNormals
    
    print(newPoints.shape)


if __name__ == '__main__':

    device = 'cuda'
    
    model = 'bracket'
    mesh = pv.read("stlFiles/bracket.stl")
    # meshVol = pv.read("./inputs/spiral_fish.ele")
     
    meshVertices = np.array(mesh.points)
    meshFaces = np.array(mesh.faces).reshape((-1,4))
    meshFaces = meshFaces[:,1:4]
    
    
    x_min,y_min,z_min = np.min(meshVertices, axis=0)
    x_max,y_max,z_max = np.max(meshVertices, axis=0)
    
    # Grid sampling - adaptive to model size
    max_range = np.max([x_max-x_min, y_max-y_min, z_max-z_min])
    ext = max_range * 0.5  # extend by half the model size
    step = max_range / 20  # ~20 points per dimension = ~8000 grid points
    x_vals = np.arange(x_min-ext, x_max+ext, step)
    y_vals = np.arange(y_min-ext, y_max+ext, step)
    z_vals = np.arange(z_min-ext, z_max+ext, step)
    
    X,Y,Z = np.meshgrid(x_vals, y_vals, z_vals)
    
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    
    
    meshGrid = pv.StructuredGrid(X,Y,Z)
            
    sdf = igl.signed_distance(meshGrid.points, meshVertices, meshFaces)[0]
    min_val = np.min([x_min, y_min, z_min])
    max_val = np.max([x_max, y_max, z_max])
    minVals = torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device = device)
    maxVals = torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device = device)
    midVals = 0.5*(minVals+maxVals)
    
    max_range = np.max([x_max-x_min,y_max-y_min, z_max-z_min])
    rangeVals = 0.5*torch.tensor([max_range,max_range,max_range], dtype=torch.float32, device=device) 
            
    
    # --- Near-surface sampling: add points INSIDE and OUTSIDE the mesh ---
    mesh.compute_normals(cell_normals=False, inplace=True)
    surfPts = np.array(mesh.points)
    surfNormals = np.array(mesh.point_data['Normals'])
    surfNormals = surfNormals / (np.linalg.norm(surfNormals, axis=1, keepdims=True) + 1e-8)
    
    near_surf_pts_list = []
    near_surf_sdf_list = []
    # Two offsets: one inside, one outside
    for offset in [-1.0, 1.0]:
        offset_pts = surfPts + offset * surfNormals
        near_surf_pts_list.append(offset_pts)
        near_surf_sdf_list.append(np.full(len(surfPts), offset))
    
    near_surf_pts = np.concatenate(near_surf_pts_list, axis=0)
    near_surf_sdf = np.concatenate(near_surf_sdf_list, axis=0)
    
    # Subsample to keep dataset size manageable
    max_near_surf = 20000
    if len(near_surf_pts) > max_near_surf:
        idx = np.random.permutation(len(near_surf_pts))[:max_near_surf]
        near_surf_pts = near_surf_pts[idx]
        near_surf_sdf = near_surf_sdf[idx]
    
    print(f"[Sampling] Grid points: {len(meshGrid.points)}, Near-surface points: {len(near_surf_pts)}")
    print(f"[Sampling] Near-surf SDF: inside={(near_surf_sdf < 0).sum()}, outside={(near_surf_sdf > 0).sum()}")
    
    # Combine grid + near-surface points
    all_points = np.concatenate([meshGrid.points, near_surf_pts], axis=0)
    all_sdf = np.concatenate([sdf, near_surf_sdf], axis=0)
    
    input_points_distribted = torch.tensor(all_points, dtype=torch.float32, device=device)
    input_scalars_distribted = torch.tensor(all_sdf/(0.5*max_range), dtype=torch.float32, device=device)
    
    
    input_points_distribted = (input_points_distribted - midVals)/rangeVals
    
    # Shuffle BEFORE slicing to avoid spatial bias
    shuffle_idx = torch.randperm(input_points_distribted.shape[0])
    input_points_distribted = input_points_distribted[shuffle_idx].detach()
    input_scalars_distribted = input_scalars_distribted[shuffle_idx].detach()
    input_points_distribted.requires_grad = True
    
    print(f"[Sampling] Total combined: {len(all_points)}, using first 150000")
    n_neg = (input_scalars_distribted[:150000] < 0).sum().item()
    n_pos = (input_scalars_distribted[:150000] > 0).sum().item()
    print(f"[Sampling] In training slice: inside={n_neg}, outside={n_pos}")
    
    
    
    
    mesh.compute_normals(cell_normals=False, inplace=True)
    boundaryPoints = mesh.points 
    boundaryNormals = mesh.point_data['Normals']
    
    
    input_points_surf = torch.tensor(boundaryPoints, dtype=torch.float32, device=device)
    input_normals_surf = torch.tensor(boundaryNormals, dtype=torch.float32, device=device)
    input_normals_surf = input_normals_surf/torch.norm(input_normals_surf, dim=1, keepdim=True)
    input_points_surf = (input_points_surf - midVals)/rangeVals
    input_points_surf = input_points_surf.detach()
    input_points_surf.requires_grad = True
    
    ###################################
    ###################################
    # newModel = sdfModel()
    newModel = sdfModel(model_load_path = "stlFiles/bracketSDF.pt")
    newModel.train(input_points_distribted[0:150000], input_scalars_distribted[0:150000], input_points_surf, input_normals_surf, savePath='stlFiles/bracketSDF.pt', epochNum=10000, lr=1e-8)

    # newModel = sdfModel(model_load_path = "./spiralSDF3.pt")
    ###################################
    ###################################
    
    
