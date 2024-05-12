clc
clear all
global Train_data class_num attr_num class_probability class_index attribute_probability attribute_index weight_L2
global raw_paraments knn_paraments spode_paraments mult_paraments t1 t2 t3 k_raw k_knn k_spode k_mult
fileName=importdata('fileName.mat');

for q=1:size(fileName,2)
%     try
    data_name=fileName{q};
    if exist(['./',data_name])
        data_name
        
        %记录每组数据预测的精度
        P_r=[];
        if exist(['./',data_name,'/paraments'])
           rmdir(['./',data_name,'/paraments'],'s');
           mkdir(['./',data_name,'/paraments']);
        else
            mkdir(['./',data_name,'/paraments'])
        end
        para=4;

        for i=1:10
            %原始数据
            raw_train=importdata(['./',data_name,'/raw_view/train_chi_',num2str(i),'.mat']);
            raw_varify=importdata(['./',data_name,'/raw_view/varify_chi_',num2str(i),'.mat']);
            raw_test=importdata(['./',data_name,'/raw_view/test_chi_',num2str(i),'.mat']);
            raw_weight_L2=importdata(['./',data_name,'/raw_mi_',num2str(i),'.mat']);
            %原始数据学习
            raw_class_num=length(unique(raw_train(:,1)));   %获取原始数据的类
            raw_attr_num=size(raw_train,2)-1;               %获取属性的数量

            %初始化矩阵
            raw_learn_weight=ones(raw_class_num*raw_attr_num,1);    %构建权重矩阵
            %权重矩阵的上界
            raw_weight_l=ones(raw_class_num*raw_attr_num,1);
            %权重矩阵的下界
            raw_weight_u=zeros(raw_class_num*raw_attr_num,1);
            %预计算贝叶斯数据
            [raw_class_probability,raw_class_index,raw_attribute_probability,raw_attribute_index]=Precomputation(raw_train);
            %预设置参数
            opts = struct('display',true,'xhistory',true,'max_iters',para);  

            %-------------------------------------------------------------------------------------------------------------

            %-------------------------------------------------------------------------------------------------------------

            %knn数据
            knn_train=importdata(['./',data_name,'/knn_view/knn_train_chi_',num2str(i),'.mat']);
            knn_varify=importdata(['./',data_name,'/knn_view/knn_varify_chi_',num2str(i),'.mat']);
            knn_test=importdata(['./',data_name,'/knn_view/knn_test_chi_',num2str(i),'.mat']);
            knn_weight_L2=importdata(['./',data_name,'/knn_mi_',num2str(i),'.mat']);

            %原始数据学习
            knn_class_num=length(unique(knn_train(:,1)));   %获取原始数据的类
            knn_attr_num=size(knn_train,2)-1;               %获取属性的数量
            %初始化矩阵
            knn_learn_weight=ones(knn_class_num*knn_attr_num,1);    %构建权重矩阵
            %权重矩阵的上界
            knn_weight_l=ones(knn_class_num*knn_attr_num,1);
            %权重矩阵的下界
            knn_weight_u=zeros(knn_class_num*knn_attr_num,1);

            %预计算贝叶斯数据
            [knn_class_probability,knn_class_index,knn_attribute_probability,knn_attribute_index]=Precomputation(knn_train);

            %预设置参数
            opts = struct('display',true,'xhistory',true,'max_iters',para);  

            %-------------------------------------------------------------------------------------------------------------

            %-------------------------------------------------------------------------------------------------------------

            %spode数据
            spode_train=importdata(['./',data_name,'/spode_view/spode_train_chi_',num2str(i),'.mat']);
            spode_varify=importdata(['./',data_name,'/spode_view/spode_varify_chi_',num2str(i),'.mat']);
            spode_test=importdata(['./',data_name,'/spode_view/spode_test_chi_',num2str(i),'.mat']);
            spode_weight_L2=importdata(['./',data_name,'/spode_mi_',num2str(i),'.mat']);

            %原始数据学习
            spode_class_num=length(unique(spode_train(:,1)));   %获取原始数据的类
            spode_attr_num=size(spode_train,2)-1;               %获取属性的数量
            %初始化矩阵
            spode_learn_weight=ones(spode_class_num*spode_attr_num,1);    %构建权重矩阵
            %权重矩阵的上界
            spode_weight_l=ones(spode_class_num*spode_attr_num,1);
            %权重矩阵的下界
            spode_weight_u=zeros(spode_class_num*spode_attr_num,1);

            %预计算贝叶斯数据
            [spode_class_probability,spode_class_index,spode_attribute_probability,spode_attribute_index]=Precomputation(spode_train);

            %预设置参数
            opts = struct('display',true,'xhistory',true,'max_iters',para);  

            %-------------------------------------------------------------------------------------------------------------

            %-------------------------------------------------------------------------------------------------------------


            %联合视图
            mult_train=importdata(['./',data_name,'/mult_view/mult_train_chi_',num2str(i),'.mat']);
            mult_varify=importdata(['./',data_name,'/mult_view/mult_varify_chi_',num2str(i),'.mat']);
            mult_test=importdata(['./',data_name,'/mult_view/mult_test_chi_',num2str(i),'.mat']);
            mult_weight_L2=importdata(['./',data_name,'/mult_mi_',num2str(i),'.mat']);

            %原始数据学习
            mult_class_num=length(unique(mult_train(:,1)));   %获取原始数据的类
            mult_attr_num=size(mult_train,2)-1;               %获取属性的数量
            %初始化矩阵
            mult_learn_weight=ones(mult_class_num*mult_attr_num,1);    %构建权重矩阵
            %权重矩阵的上界
            mult_weight_l=ones(mult_class_num*mult_attr_num,1);
            %权重矩阵的下界
            mult_weight_u=zeros(mult_class_num*mult_attr_num,1);

            %预计算贝叶斯数据
            [mult_class_probability,mult_class_index,mult_attribute_probability,mult_attribute_index]=Precomputation(mult_train);

            %预设置参数
            opts = struct('display',true,'xhistory',true,'max_iters',para);  

            %-------------------------------------------------------------------------------------------------------------
            %预设置 
            %初始化最终的矩阵参数
            combine_learn_weight=ones(mult_class_num*mult_attr_num,1);    %构建权重矩阵
            %权重矩阵的上界
            combine_weight_l=ones(mult_class_num*mult_attr_num,1);
            %权重矩阵的下界
            combine_weight_u=zeros(mult_class_num*mult_attr_num,1);


            raw_paraments=cell(1,8);
            knn_paraments=cell(1,8);
            spode_paraments=cell(1,8);
            mult_paraments=cell(1,8);
            raw_paraments={raw_train,raw_class_num,raw_attr_num,raw_class_probability,raw_class_index,raw_attribute_probability,raw_attribute_index,raw_weight_L2};
            knn_paraments={knn_train,knn_class_num,knn_attr_num,knn_class_probability,knn_class_index,knn_attribute_probability,knn_attribute_index,knn_weight_L2};
            spode_paraments={spode_train,spode_class_num,spode_attr_num,spode_class_probability,spode_class_index,spode_attribute_probability,spode_attribute_index,spode_weight_L2};
            mult_paraments={mult_train,mult_class_num,mult_attr_num,mult_class_probability,mult_class_index,mult_attribute_probability,mult_attribute_index,mult_weight_L2};

            t1=raw_class_num*raw_attr_num;          %原始数据
            t2=knn_class_num*knn_attr_num;          %knn数据
            t3=spode_class_num*spode_attr_num;      %spode数据


            %-----------------------
            %迭代次数 
            N=21;
            %参数的记录
            k_parament=zeros(4,N-1);
            combine_x_history_paraments=[];
            combine_f_history_paraments=[];
            T1_f=[];
            T2_f=[];
            T3_f=[];
            T4_f=[];
            T1_v=[];
            T2_v=[];
            T3_v=[];
            T4_v=[];

            %-----------------------

            %加入循环
            F_L=0;
            j=1;
            while(j<N)
            
                j

                %每个视图的参数更新

                raw_learn_weight=combine_learn_weight(1:t1,1);
                knn_learn_weight=combine_learn_weight(t1+1:t1+t2,1);
                spode_learn_weight=combine_learn_weight(t1+t2+1:t1+t2+t3,1);
                mult_learn_weight=combine_learn_weight(:,1);




                %（1）原始数据
                %-----------------------------------------------
                %参数的临时传递
                Train_data=raw_train;
                class_num=raw_class_num;
                attr_num=raw_attr_num;
                class_probability=raw_class_probability; 
                class_index=raw_class_index;
                attribute_probability=raw_attribute_probability; 
                attribute_index=raw_attribute_index;
                weight_L2=raw_weight_L2;

                %
                func = @(W) rosenbrock(W);

                [raw_x,raw_xhistory,raw_fhistory] = LBFGSB(func,raw_learn_weight,raw_weight_u,raw_weight_l,opts);  
                
                if numel(find(isnan(raw_x)))
                    raw_x=raw_xhistory(:,end-1)
                    break
                end
                 %-----------------------------------------------
                  %（2）knn数据
                %-----------------------------------------------
                %参数的临时传递
                Train_data=knn_train;
                class_num=knn_class_num;
                attr_num=knn_attr_num;
                class_probability=knn_class_probability; 
                class_index=knn_class_index;
                attribute_probability=knn_attribute_probability; 
                attribute_index=knn_attribute_index;
                weight_L2=knn_weight_L2;    
                func = @(W) rosenbrock(W);

                [knn_x,knn_xhistory,knn_fhistory] = LBFGSB(func,knn_learn_weight,knn_weight_u,knn_weight_l,opts);
                 if numel(find(isnan(knn_x)))
                    break
                end
                  %-----------------------------------------------
                   %（3）spode数据
                %-----------------------------------------------
                %参数的临时传递
                Train_data=spode_train;
                class_num=spode_class_num;
                attr_num=spode_attr_num;
                class_probability=spode_class_probability; 
                class_index=spode_class_index;
                attribute_probability=spode_attribute_probability; 
                attribute_index=spode_attribute_index;
                weight_L2=spode_weight_L2;    
                func = @(W) rosenbrock(W); 

                [spode_x,spode_xhistory,spode_fhistory] = LBFGSB(func,spode_learn_weight,spode_weight_u,spode_weight_l,opts); 
                if numel(find(isnan(spode_x)))
                    break
                end
                   %-----------------------------------------------
                   
                    %（4）mult数据
                %-----------------------------------------------  
                %参数的临时传递
                Train_data=mult_train;
                class_num=mult_class_num;
                attr_num=mult_attr_num;
                class_probability=mult_class_probability; 
                class_index=mult_class_index;
                attribute_probability=mult_attribute_probability; 
                attribute_index=mult_attribute_index;
                weight_L2=mult_weight_L2;   
                func = @(W) rosenbrock(W);   

                [mult_x,mult_xhistory,mult_fhistory] = LBFGSB(func,mult_learn_weight,mult_weight_u,mult_weight_l,opts);  
                  if numel(find(isnan(mult_x)))
                        break
                   end
                   %-----------------------------------------------
                   
                Train_data=raw_train;
                class_num=raw_class_num;
                attr_num=raw_attr_num;
                class_probability=raw_class_probability; 
                class_index=raw_class_index;
                attribute_probability=raw_attribute_probability; 
                attribute_index=raw_attribute_index;
                weight_L2=raw_weight_L2;
                raw_varify_fhistory=zeros(1,size(raw_fhistory,2));
                %计算验证集损失
                for k=1:size(raw_xhistory,2)
                    raw_varify_fhistory(1,k)=Verify_loss(raw_xhistory(:,k),raw_varify);
                end

                
                [raw_varify_pre]=GetPre(raw_varify,raw_x);
                





                norm_raw_fhistory=raw_fhistory/size(raw_train,1);
                norm_raw_varify_fhistory=raw_varify_fhistory/size(raw_varify,1);
                
                 T1_f=[T1_f,norm_raw_fhistory];
                 T1_v=[T1_v,norm_raw_varify_fhistory];
                 

                delta_raw=((norm_raw_fhistory(end)-norm_raw_fhistory(1))-(norm_raw_varify_fhistory(end)-norm_raw_varify_fhistory(1)))^2;
                H_raw=(norm_raw_varify_fhistory(end)/norm_raw_varify_fhistory(1))^2;



               

                Train_data=knn_train;
                class_num=knn_class_num;
                attr_num=knn_attr_num;
                class_probability=knn_class_probability; 
                class_index=knn_class_index;
                attribute_probability=knn_attribute_probability; 
                attribute_index=knn_attribute_index;
                weight_L2=knn_weight_L2;  
                knn_varify_fhistory=zeros(1,size(knn_fhistory,2));
                %计算验证集损失
                for k=1:size(knn_xhistory,2)
                    knn_varify_fhistory(1,k)=Verify_loss(knn_xhistory(:,k),knn_varify);
                end    

                [knn_varify_pre]=GetPre(knn_varify,knn_x);

                norm_knn_fhistory=knn_fhistory/size(knn_train,1);
                norm_knn_varify_fhistory=knn_varify_fhistory/size(knn_varify,1);
                
                T2_f=[T2_f,norm_knn_fhistory];
                T2_v=[T2_v,norm_knn_varify_fhistory];
                
                delta_knn=((norm_knn_fhistory(end)-norm_knn_fhistory(1))-(norm_knn_varify_fhistory(end)-norm_knn_varify_fhistory(1)))^2;
                H_knn=(norm_knn_varify_fhistory(end)/norm_knn_varify_fhistory(1))^2;

             



                Train_data=spode_train;
                class_num=spode_class_num;
                attr_num=spode_attr_num;
                class_probability=spode_class_probability; 
                class_index=spode_class_index;
                attribute_probability=spode_attribute_probability; 
                attribute_index=spode_attribute_index;
                weight_L2=spode_weight_L2;  

                spode_varify_fhistory=zeros(1,size(spode_fhistory,2));
                %计算验证集损失
                for k=1:size(spode_xhistory,2)
                    spode_varify_fhistory(1,k)=Verify_loss(spode_xhistory(:,k),spode_varify);
                end    

                [spode_varify_pre]=GetPre(spode_varify,spode_x);

                norm_spode_fhistory=spode_fhistory/size(spode_train,1);
                norm_spode_varify_fhistory=spode_varify_fhistory/size(spode_varify,1);
                
                T3_f=[T3_f,norm_spode_fhistory];
                T3_v=[T3_v,norm_spode_varify_fhistory];
                delta_spode=((norm_spode_fhistory(end)-norm_spode_fhistory(1))-(norm_spode_varify_fhistory(end)-norm_spode_varify_fhistory(1)))^2;
                H_spode=(norm_spode_varify_fhistory(end)/norm_spode_varify_fhistory(1))^2;


               
                Train_data=mult_train;
                class_num=mult_class_num;
                attr_num=mult_attr_num;
                class_probability=mult_class_probability; 
                class_index=mult_class_index;
                attribute_probability=mult_attribute_probability; 
                attribute_index=mult_attribute_index;
                weight_L2=mult_weight_L2;   

                mult_varify_fhistory=zeros(1,size(mult_fhistory,2));
                %计算验证集损失
                for k=1:size(mult_xhistory,2)
                    mult_varify_fhistory(1,k)=Verify_loss(mult_xhistory(:,k),mult_varify);
                end    

                [mult_varify_pre]=GetPre(mult_varify,mult_x);

                norm_mult_fhistory=mult_fhistory/size(mult_train,1);
                norm_mult_varify_fhistory=mult_varify_fhistory/size(mult_varify,1);
                
                T4_f=[T4_f,norm_mult_fhistory];
                T4_v=[T4_v,norm_mult_varify_fhistory];

                delta_mult=((norm_mult_fhistory(end)-norm_mult_fhistory(1))-(norm_mult_varify_fhistory(end)-norm_mult_varify_fhistory(1)))^2;

                H_mult=(norm_mult_varify_fhistory(end)/norm_mult_varify_fhistory(1))^2;



                norm_delta_raw=delta_raw;
                norm_delta_knn=delta_knn;
                norm_delta_spode=delta_spode;
                norm_delta_mult=delta_mult;

                norm_H_raw=H_raw;
                norm_H_knn=H_knn;
                norm_H_spode=H_spode;
                norm_H_mult=H_mult;
                
                %对公式进行计算
                temp_raw=1/(2*(norm_delta_raw+norm_H_raw));
                temp_knn=1/(2*(norm_delta_knn+norm_H_knn));
                temp_spode=1/(2*(norm_delta_spode+norm_H_spode));
                temp_mult=1/(2*(norm_delta_mult+norm_H_mult));

                Nemta=1/(temp_raw+temp_knn+temp_spode+temp_mult);

                k_raw=Nemta*temp_raw;
                k_knn=Nemta*temp_knn;
                k_spode=Nemta*temp_spode;
                k_mult=Nemta*temp_mult;


                %融入4个精度的差异
                %raw_varify_pre，knn_varify_pre，spode_varify_pre，mult_varify_pre

                k_parament(:,j)=[k_raw;k_knn;k_spode;k_mult];

                all=k_raw+k_knn+k_spode+k_mult;


                %将每个模态的数据进行联合训练
                %上面的k值是对与数据的预估，所生成的权重

                %[f,g]=combine_gradient(combine_learn_weight);

                %进行联合学习
                func = @(W) combine_gradient(W); 

                [combine_x,combine_xhistory,combine_fhistory] = LBFGSB(func,combine_learn_weight,combine_weight_u,combine_weight_l,opts);  
                
                 if numel(find(isnan(combine_x)))
                     if size(combine_xhistory,2)==1
                         combine_learn_weight=combine_x_history_paraments(:,end);
                          break
                     else
                         combine_xhistory(:,end)=[];
                         combine_fhistory(end)=[];
                         combine_x=combine_xhistory(:,end);
                         combine_learn_weight=combine_x;

                         combine_x_history_paraments=[combine_x_history_paraments,combine_xhistory];

                         combine_f_history_paraments=[combine_f_history_paraments,combine_fhistory];
                         
                         break
                     end
                       
                 end

                combine_learn_weight=combine_x;

                combine_x_history_paraments=[combine_x_history_paraments,combine_xhistory];

                combine_f_history_paraments=[combine_f_history_paraments,combine_fhistory];



                j=j+1;
           
                
    
            


            end



            save(['./',data_name,'/paraments/combine_x_history_paraments_',int2str(i)],'combine_x_history_paraments');
            save(['./',data_name,'/paraments/combine_f_history_paraments_',int2str(i)],'combine_f_history_paraments');
            save(['./',data_name,'/paraments/T1_f_',int2str(i)],'T1_f');
            save(['./',data_name,'/paraments/T1_v_',int2str(i)],'T1_v');
            save(['./',data_name,'/paraments/T2_f_',int2str(i)],'T2_f');
            save(['./',data_name,'/paraments/T2_v_',int2str(i)],'T2_v');
            save(['./',data_name,'/paraments/T3_f_',int2str(i)],'T3_f');
            save(['./',data_name,'/paraments/T3_v_',int2str(i)],'T3_v');
            save(['./',data_name,'/paraments/T4_f_',int2str(i)],'T4_f');
            save(['./',data_name,'/paraments/T4_v_',int2str(i)],'T4_v');
            save(['./',data_name,'/paraments/k_parament_',int2str(i)],'k_parament');
            

            raw_learn_weight=combine_learn_weight(1:t1,1);
            knn_learn_weight=combine_learn_weight(t1+1:t1+t2,1);
            spode_learn_weight=combine_learn_weight(t1+t2+1:t1+t2+t3,1);
            mult_learn_weight=combine_learn_weight(:,1);

            %针对获取的的combine_learn_weight进行数据的分析
            [Train_data,class_num,attr_num,class_probability,class_index,attribute_probability,attribute_index,weight_L2]=raw_paraments{1,:};
            raw_learn_weight=combine_learn_weight(1:t1,1);
            raw_precision=ResultNB(raw_test,raw_learn_weight);

            [Train_data,class_num,attr_num,class_probability,class_index,attribute_probability,attribute_index,weight_L2]=knn_paraments{1,:};
            knn_learn_weight=combine_learn_weight(t1+1:t1+t2,1);
            knn_precision=ResultNB(knn_test,knn_learn_weight);

            [Train_data,class_num,attr_num,class_probability,class_index,attribute_probability,attribute_index,weight_L2]=spode_paraments{1,:};
            spode_learn_weight=combine_learn_weight(t1+t2+1:t1+t2+t3,1);
            spode_precision=ResultNB(spode_test,spode_learn_weight);

            [Train_data,class_num,attr_num,class_probability,class_index,attribute_probability,attribute_index,weight_L2]=mult_paraments{1,:};
            mult_learn_weight=combine_learn_weight;
            mult_precision=ResultNB(mult_test,mult_learn_weight);
            
            
            
            
            

            Result_precisiom=k_raw*raw_precision+k_knn*knn_precision+k_spode*spode_precision+k_mult*mult_precision;

            Result_pre=zeros(size(Result_precisiom,1),1);
            for k=1:size(Result_precisiom,1)
                TTL=find(max(Result_precisiom(k,:))==Result_precisiom(k,:));
                temp_l=TTL(1);
                clear TTL
                Result_pre(k,1)=class_index(1,temp_l);
            end
            num=length(find(Result_pre==mult_test(:,1)));
            P_r=[P_r,num/size(mult_test,1)];


        end

        save(['./',data_name,'/P_R'],'P_r');

       
    end
%     catch
%     end
end

% data_name




%联合学习后的梯度变化
function [f,g]=combine_gradient(weight)
    global raw_paraments knn_paraments spode_paraments mult_paraments
    global Train_data class_num attr_num class_probability class_index attribute_probability attribute_index weight_L2
    global t1 t2 t3 k_raw k_knn k_spode k_mult
    raw_learn_weight=weight(1:t1,1);
    [Train_data,class_num,attr_num,class_probability,class_index,attribute_probability,attribute_index,weight_L2]=raw_paraments{1,:};
    [raw_f,raw_g]=rosenbrock(raw_learn_weight);

    knn_learn_weight=weight(t1+1:t1+t2,1);
    [Train_data,class_num,attr_num,class_probability,class_index,attribute_probability,attribute_index,weight_L2]=knn_paraments{1,:};
    [knn_f,knn_g]=rosenbrock(knn_learn_weight);
    
    
    spode_learn_weight=weight(t1+t2+1:t1+t2+t3,1);
    [Train_data,class_num,attr_num,class_probability,class_index,attribute_probability,attribute_index,weight_L2]=spode_paraments{1,:};
    [spode_f,spode_g]=rosenbrock(spode_learn_weight);
    
    [Train_data,class_num,attr_num,class_probability,class_index,attribute_probability,attribute_index,weight_L2]=mult_paraments{1,:};
    [mult_f,mult_g]=rosenbrock(weight);
    
    f=raw_f*k_raw+knn_f*k_knn+spode_f*k_spode+mult_f*k_mult;
    
    %特征进行拼接
    Q=[raw_g*k_raw;knn_g*k_knn;spode_g*k_spode];
    g=Q+k_mult*mult_g;
    
    
end


%贝叶斯预计算矩阵
function [class_probability,class_index,attribute_probability,attribuye_index]=Precomputation(train)
    class_num=length(unique(train(:,1)));   %获取原始数据的类
    attr_num=size(train,2)-1;               %获取属性的数量
    %train训练集 
    [m,n]=size(train);
    %%类标签概率存储为一个 一行data_class列的数据
    class_probability=zeros(1,class_num);
    class_index=unique(train(:,1))';
    T1_fenmu=(m+1);
    parfor i=1:class_num
        num=length(find(train(:,1)==class_index(1,i)));
        class_probability(1,i)=(num+1/class_num)/T1_fenmu;
    end
   
    %构建一个元组，存放每个属性对应的标签的概率
    attribute_probability=cell(attr_num,1);
    attribuye_index=cell(attr_num,1);
    parfor i=2:n
        %查看第i个属性的取值的种类
        T1=unique(train(:,i));
        attr_species=length(T1);
        %构建临时矩阵存储概率参数
        %矩阵的行为类别数量，列为属性的取值数
        temp_matrix=zeros(class_num,attr_species);
        attr_temp=T1';
        for j=1:class_num
            [temp_class_index]=find(train(:,1)==class_index(1,j));  %查找训练集中类标签，属于第j类的标签
            T_fenmu=(length(temp_class_index)+1);
            for k=1:attr_species
                [temp_attr_index]=find(train(:,i)==attr_temp(1,k));  %查看训练集第i个属性值为k的位置
                only_attr_data=length(intersect(temp_class_index,temp_attr_index));    %属于第j类且值为k的数量
                temp_matrix(j,k)=(only_attr_data+1/attr_species)/T_fenmu;
            end
        end
        attribute_probability{i-1,1}=temp_matrix;       %行表示类别、列表示属性的取值
        attribuye_index{i-1,1}=attr_temp;

    end
end



function [f,g]=rosenbrock(weight)
    global Train_data class_num attr_num class_probability class_index attribute_probability attribute_index weight_L2
    
    %将矩阵进行重新构建
    new_weight=reshape(weight,class_num,attr_num);
    more_weight=repmat(weight_L2,class_num,1);
    %初始化函数值
    f=0;
    
    [m,n]=size(Train_data);
    
    PrecisionResult=ResultNB(Train_data,weight);
    
    parfor i=1:m
       corr_index=find(class_index==Train_data(i,1));
       for j=1:class_num
           if j==corr_index
               f=f+0.5*(1-PrecisionResult(i,j))^2;
           else
               f=f+0.5*(0-PrecisionResult(i,j))^2;
           end
       end
    end
    
    %此处添加范式
    cha=new_weight-more_weight;
    cha=cha.^2;
    f=f+0.5*sum(sum(cha));
    
    if (nargout > 1)
    g=zeros(class_num*attr_num,1);
    Temp_w=reshape(1:class_num*attr_num,class_num,attr_num);
        %求导每一个数据
        parfor p=1:(class_num*attr_num)
            [x,y]=find(Temp_w==p);  %x表示行、y表示列
            for i=1:m
                corr_index=find(class_index==Train_data(i,1));
                matrix=attribute_probability{y};
                matrix_index=attribute_index{y};      

                iidd=find(matrix_index(1,:)==Train_data(i,y+1));
                for k=1:class_num
                    if k==corr_index
                        if x==k
                            g(p)=g(p)-(1-PrecisionResult(i,k))*PrecisionResult(i,k)*(1-PrecisionResult(i,x))*log(matrix(x,iidd));
                        else
                            g(p)=g(p)+(1-PrecisionResult(i,k))*PrecisionResult(i,k)*PrecisionResult(i,x)*log(matrix(x,iidd));
                        end

                    else
                        if x==k
                            g(p)=g(p)-(0-PrecisionResult(i,k))*PrecisionResult(i,k)*(1-PrecisionResult(i,x))*log(matrix(x,iidd));
                        else
                            g(p)=g(p)+(0-PrecisionResult(i,k))*PrecisionResult(i,k)*PrecisionResult(i,x)*log(matrix(x,iidd));
                        end 

                    end
                end
            end  
            
            %此处添加范式
            g(p)=g(p)+(new_weight(x,y)-more_weight(x,y));
        end
    
    end
    
end


function PrecisionResult=ResultNB(train,weight)
    
    global class_num attr_num class_probability class_index attribute_probability attribute_index
    
    %矩阵重构
    normal_weight=reshape(weight,class_num,attr_num);
    
    test=train;
    
    [m,n]=size(train);

    [a,b]=size(test);
    
    PrecisionResult=zeros(a,class_num);
    
    parfor i=1:a
        %遍历每一个数据集
        temp_class_pre=zeros(1,class_num);     %这个数据被预测为哪个类别
        
        for j=1:class_num
            
            temp_pre=class_probability(1,j);
            
            %此时的标签为class_index(1,j)
            
            for k=2:b
                
                %k表示第几个属性
                temp=test(i,k);
                
                %拿出对应属性的参考矩阵
           
                
                matrix=attribute_probability{k-1};
                
                matrix_index=attribute_index{k-1};
                
                if ~isempty(find(matrix_index(1,:)==temp, 1))
                    
                    lloc=matrix_index(1,:)==temp;
                    
                    temp_pre=temp_pre*(matrix(j,lloc)^(normal_weight(j,k-1)));
                    
                else
                    
%                     fprintf("数据划分有误\n");
                    
                end
                
       
            end
            
            temp_class_pre(1,j)=temp_pre;
     
        end
        
        PrecisionResult(i,:)=temp_class_pre/sum(temp_class_pre(1,:));

    end
    
end


function [f]=Verify_loss(weight,verify)
    global   class_num attr_num class_probability class_index attribute_probability attribute_index weight_L2
    
    more_weight=repmat(weight_L2,class_num,1);
        
    normal_weight=reshape(weight,class_num,attr_num);
    
    %初始化函数值
    f=0;
    
    PrecisionResult=ResultNB(verify,weight);
    
    [m,n]=size(verify);
    
    parfor i=1:m
       corr_index=find(class_index== verify (i,1));
       for j=1:class_num
           if j==corr_index
               f=f+0.5*(1-PrecisionResult(i,j))^2;
           else
               f=f+0.5*(0-PrecisionResult(i,j))^2;
           end
       end
    end  
    
    %此处添加范式
    cha=normal_weight-more_weight;
    cha=cha.^2;
    f=f+0.5*sum(sum(cha));
    
end


function [raw_varify_pre]=GetPre(raw_varify,raw_x)
    raw_Precision=ResultNB(raw_varify,raw_x);
    raw_pre=zeros(size(raw_Precision,1),1);
    for i=1:size(raw_Precision,1)
        [a,b]=max(raw_Precision(i,:));
        raw_pre(i,1)=b;
    end
    temp_raw1=raw_pre-raw_varify(:,1);
    raw_varify_pre=length(find(temp_raw1==0))/size(raw_varify,1);
end





