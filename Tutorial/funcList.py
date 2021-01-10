# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## List of functions

# %%
import numpy as np
from scipy.optimize import curve_fit, minimize
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import cm

# %% [markdown]
# ### Extract and process data
# The functions below are used to extract and process the data

# %%


def extractData(path, colX="", colY="", colZ="", colList=[]):
    """
    Return the dataframe inside the file in the path with specified column name
    Assumption: All column has head
    Example
    >>> extractData('dir/file.csv','X','Y','Z')
    >>> extractData('dir/file.csv','X','Y')
    >>> extractData('dir/file.csv')
    if colX is not specified all columns will be extracted
    """
    if(len(colList) >= 1):
        data1 = pd.read_csv(path, usecols=colList)
    elif(colX != "" and colY != ""):
        colList = [colX, colY]
        if(colZ != ""):
            colList.append(colZ)
        data1 = pd.read_csv(path, usecols=colList)
    else:
        data1 = pd.read_csv(path)
    if(len(colList) >= 1):
        if(isinstance(colList[0], int)):
            colList2 = colList.copy()
            colList2.sort()
            colList = [colList2.index(col) for col in colList]
            data1 = data1.iloc[:, colList]
        else:
            data1 = data1[colList]
    return data1


def isFloat(num):
    """
    Check if input is float
    """
    str1 = str(num).lower()
    try:
        float(str1)
    except ValueError:
        return False
    return True and (str1 != 'nan')


def cleanData(df):
    """
    Clean the dataframe such that all row with value that is not float will be deleted. Convert all row to float
    """
    return df[df.applymap(isFloat).apply(all, axis=1)].applymap(lambda x: float(x))


def linFun(x, a, b):
    """
    Linear function
    """
    return a*x+b


def gaussFun(x, A, w, xc, y0):
    """
    Gaussian peak function. 
    Note: A = Amplitude, w = FWHM, xc=center, y0=offset
    """
    w = w/(2*np.sqrt(np.log(4)))  # change FWHM to sigma
    return y0+A*np.exp(-((x-xc)**2)/(2*(w**2)))


def gauss2Fun(x, A1, w1, xc1, A2, w2, xc2, y0):
    """
    Double Gaussian peak function.
    Note: A = Amplitude, w = FWHM, xc=center, y0=offset
    """
    return gaussFun(x, A1, w1, xc1, y0)+gaussFun(x, A2, w2, xc2, 0)


def lorentzFun(x, A, w, xc, y0):
    """
    Lorentzian peak function
    Note: A = Amplitude, w = FWHM, xc=center, y0=offset
    """
    return y0+A*(w**2)/(4*(x-xc)**2+w**2)


def lorentz2Fun(x, A1, w1, xc1, A2, w2, xc2, y0):
    """
    Double Lorentzian peak function.
    Note: A = Amplitude, w = FWHM, xc=center, y0=offset
    """
    return lorentzFun(x, A1, w1, xc1, y0)+lorentzFun(x, A2, w2, xc2, 0)


def myFit(fun, x, y):
    """
    Curve fitting returning parameter values and errors
    """
    fitParam, fitErr = curve_fit(fun, x, y)
    fitErr = [np.sqrt(fitErr[i][i]) for i in range(len(fitErr))]
    y_arr = np.array(y)
    x_arr = np.array(x)
    residual = y_arr-fun(x_arr, *fitParam)
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_arr-np.mean(y_arr))**2)
    r_squared = 1-(ss_res/ss_tot)
    return fitParam.tolist()+fitErr+[r_squared]


def singleFit(df, fun, p0=None, bUp=None, bLow=None, colX=0, colY=1, ax=None, typeList=['auto0;;auto0'], labelList=['Fit'], multPoint=10, fitList=None, plotData=True, showPlot=True):
    """
    Curve fitting returning parameter values and errors + plotting the fitting graph together with data if requested. Axis object must be provided in the latter case.
    """
    x = df.iloc[:, colX]
    y = df.iloc[:, colY]
    if(fitList == None):
        curveParam = {}
        if(p0 != None):
            curveParam['p0'] = p0
        boundVal = [-np.inf, np.inf]
        if(bLow != None):
            boundVal[0] = bLow
        if(bUp != None):
            boundVal[1] = bUp
        curveParam['bounds'] = tuple(boundVal)
        fitParam, fitErr = curve_fit(fun, x, y, **curveParam)
        fitParam = fitParam.tolist()
        # the default is 68% confidence interval=1 sigma. For 95% (the one MATLAB used) times this by 2
        fitErr = [np.sqrt(fitErr[i][i]) for i in range(len(fitErr))]
        y_arr = np.array(y)
        x_arr = np.array(x)
        residual = y_arr-fun(x_arr, *fitParam)
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((y_arr-np.mean(y_arr))**2)
        r_squared = 1-(ss_res/ss_tot)
        fitList = fitParam+fitErr+[r_squared]
    else:
        fitParam = fitList[0:int((len(fitList)-1)/2)]
    if(ax != None):
        xMax = max(x)
        xMin = min(x)
        x2 = list(np.linspace(start=xMin, stop=xMax, num=multPoint*len(x)))
        y2 = [fun(xVal, *fitParam) for xVal in x2]
        dfList = [pd.DataFrame({x.name: x2, y.name: y2})]
        markerEdgeWidth = 3
        if(plotData):
            dfList = dfList+[df.iloc[:, [colX, colY]]]
            markerEdgeWidth = 1
        plot1Graph(dataList=dfList, typeList=typeList,
                   labelList=labelList, axObj=ax, markerEdgeWidth=markerEdgeWidth, isPlot=showPlot)
    return fitList


def multiFit(colX, colY, fun, p0=None, bUp=None, bLow=None, dfg=None, dfd=None, axList=[], axPrimer=[], paramList=[], xLabel=[], yLabel=[], xyAx=None, valList=[], plotData=True, plotErrType=None, plotType='marker', showPlot=True):
    """
    Multiple data set fitting. The input is either a dictionary of dataframe (dfd) or the dataframegroupby (dfg). If the axisList is provided the fitting parameter and its error can be plotted provided the label in dictionary or groupby can be converted to float. If xyAx is provided, the individual datasets together with their fitted graph can be plotted. 
    """
    inType = 'None'
    if(dfg != None):
        inType = 'groupBy'
        dfFit = dfg.apply(lambda df1: singleFit(
            df=df1.iloc[:, [colX, colY]], fun=fun))
    elif(dfd != None):
        dfFit = pd.Series(
            {k: singleFit(fun=linFun, df=v.iloc[:, [colX, colY]]) for k, v in dfd.items()})
        inType = 'dict'
    else:
        return None
    parExLen = len(dfFit.iloc[0])-1
    axLen = min(len(axList), int(parExLen/2))
    axList = axList[:axLen]
    if(axLen > 0):
        if(len(paramList) == 0):
            paramList = list(range(axLen))
        if(len(xLabel) == 0):
            xLabel = ['x']*axLen
        if(len(yLabel) == 0):
            yLabel = ['y']*axLen
        loopNum = min(len(paramList), axLen)
        xLabel = xLabel+[xLabel[0]]*max(0, loopNum-len(xLabel))
        yLabel = yLabel+[yLabel[0]]*max(0, loopNum-len(yLabel))
        if(len(axPrimer) > 1 and len(axPrimer) < loopNum):
            axPrimer = axList
        xList = list(dfFit.index)
        if(all([isFloat(x) for x in xList])):
            xList = [float(x) for x in xList]
            paramVal = pd.DataFrame.from_dict(
                dict(zip(range(len(xList)), dfFit.values)), orient='index')
            typeNum = 0
            for ind in range(loopNum):
                df = pd.DataFrame(
                    {xLabel[ind]: xList, yLabel[ind]: paramVal.iloc[:, paramList[ind]]})
                if(plotType == 'marker'):
                    typeList = ['auto'+str(typeNum) +
                                ';'+'auto'+str(typeNum)+';']
                elif(plotType == 'line'):
                    typeList = ['auto'+str(typeNum)+';;'+'auto'+str(typeNum)]
                else:
                    typeList = [
                        'auto'+str(typeNum)+';'+'auto'+str(typeNum)+';'+'auto'+str(typeNum)]
                plot1Graph(dataList=[df], axObj=axList[ind], labelList=[
                           yLabel[ind]], typeList=typeList, isPlot=showPlot)
                colorType = autoType('auto'+str(typeNum)+';;')
                yErr = [2*val for val in paramVal.iloc[:,
                                                       paramList[ind]+int(parExLen/2)]]  # draw 2 sigma
                if(plotErrType == 'bar'):
                    axList[ind].errorbar(x=df.iloc[:, 0], y=df.iloc[:, 1], yerr=yErr,
                                         capsize=5, capthick=2, fmt='none', ecolor=colorType)
                if(plotErrType == 'fill'):
                    axList[ind].fill_between(
                        x=df.iloc[:, 0], y1=df.iloc[:, 1]-yErr, y2=df.iloc[:, 1]+yErr, color=colorType, alpha=0.3)
                if(len(axPrimer) == loopNum):
                    createLegend(axPrimer=axList[ind])
                typeNum = typeNum+1
            if(len(axPrimer) == 1 and len(axPrimer) < loopNum):
                createLegend(axPrimer=axPrimer[0],
                             axList=axList, boxLoc=[1, 1, 1])
    if(xyAx != None):
        typeNum = 0
        for val in valList:
            typeList = ['auto'+str(typeNum)+';;'+'auto'+str(typeNum)]
            labelList = [str(val)+'Fit']
            if(plotData):
                typeList = typeList + \
                    ['auto'+str(typeNum)+';'+'auto'+str(typeNum)+';']
                labelList = labelList+[str(val)]
            if(inType == 'groupBy'):
                df = dfg.get_group(val).iloc[:, [colX, colY]]
            elif(inType == 'dict'):
                df = dfd[val].iloc[:, [colX, colY]]
            else:
                break
            singleFit(df=df, fun=fun, ax=xyAx, multPoint=1,
                      fitList=dfFit[val], typeList=typeList, labelList=labelList, plotData=plotData, showPlot=showPlot, p0=p0, bUp=bUp, bLow=bLow)
            createLegend(axPrimer=xyAx, boxLoc=[1, 1, 3])
            typeNum = typeNum+1
    return dfFit


# %% [markdown]
# ### Canvas setup
# The functions below are used to ecreate and prepare figure (with no data)

# %%
def setDefault():
    """
    Setting default setup for drawing the figure
    """
    mpl.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2


def createFig(width=3.8, height=3, title=""):
    """
    Given width and height create new figure
    >>> fig = createFig(width=4,height=3,title="tes")
    >>> fig = createFig()
    """
    fig = plt.figure(figsize=(width, height))
    if(title != ""):
        fig.suptitle(title, y=1.1)
    return fig


def createAx(fig, ax1=None, xTickLoc='', yTickLoc='', rect=[0, 0, 1, 1]):
    """
    Given fig or axis create new axis. If axis are given then depending on the xTickLoc and yTickLoc, a new axis with shared x axis, shared y axis, or all new axes will be created
    >>> createAx(fig=fig,ax1=ax,xTickLoc='bottom',yTickLoc='left',rect=[0,0,1,1])
    >>> createAx(fig=fig,ax1=ax)
    >>> createAx(fig=fig)
    """
    xTickLoc = xTickLoc.lower()
    yTickLoc = yTickLoc.lower()
    if(ax1 == None):
        ax = fig.add_axes(rect)
        # ax = fig.add_subplot(111)
        if(xTickLoc == ''):
            xTickLoc = 'bottom'
        if(yTickLoc == ''):
            yTickLoc = 'left'
    else:
        ax1_xTickLoc = ax1.xaxis.get_ticks_position()
        ax1_yTickLoc = ax1.yaxis.get_ticks_position()
        if(xTickLoc == '' or xTickLoc == ax1_xTickLoc):
            ax = ax1.twinx()
            xTickLoc = ''
            if(yTickLoc == ''):
                if(ax1_yTickLoc == 'right'):
                    yTickLoc = 'left'
                else:
                    yTickLoc = 'right'
        else:
            if(yTickLoc == '' or yTickLoc == ax1_yTickLoc):
                ax = ax1.twiny()
                yTickLoc = ''
            else:
                ax = fig.add_axes(ax1.get_position())
    if(xTickLoc == 'bottom'):
        ax.xaxis.tick_bottom()
        ax.xaxis.set_tick_params(
            which='major', size=10, width=2, direction='in', top=False, bottom=True)
        ax.xaxis.set_tick_params(
            which='minor', size=7, width=2, direction='in', top=False, bottom=True)
    if(xTickLoc == 'top'):
        ax.xaxis.tick_top()
        ax.xaxis.set_tick_params(
            which='major', size=10, width=2, direction='in', top=True, bottom=False)
        ax.xaxis.set_tick_params(
            which='minor', size=7, width=2, direction='in', top=True, bottom=False)
    if(yTickLoc == 'left'):
        ax.yaxis.tick_left()
        ax.yaxis.set_tick_params(
            which='major', size=10, width=2, direction='in', right=False, left=True)
        ax.yaxis.set_tick_params(
            which='minor', size=7, width=2, direction='in', right=False, left=True)
    if(yTickLoc == 'right'):
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(
            which='major', size=10, width=2, direction='in', right=True, left=False)
        ax.yaxis.set_tick_params(
            which='minor', size=7, width=2, direction='in', right=True, left=False)
    return ax


def setAx(ax, xLow=None, xHigh=None, yLow=None, yHigh=None, xStep=None, yStep=None, xLabel=None, yLabel=None):
    """
    Set the axis property (limit, step, label). All property other than ax is optional
    >>> setAx(ax,xLow=0,xHigh=1,yLow=0,yHigh=1,xStep=0.2,yStep=0.2,xLabel='x',yLabel='y')
    """
    if(xLow != None and xHigh != None):
        if(xLow < xHigh):
            ax.set_xlim(xLow, xHigh)
    if(yLow != None and yHigh != None):
        if(yLow < yHigh):
            ax.set_ylim(yLow, yHigh)
    if(xStep != None):
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xStep))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xStep/2))
    if(yStep != None):
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(yStep))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(yStep/2))
    if(xLabel != None):
        ax.set_xlabel(xLabel, labelpad=2)
        ax.xaxis.set_label_position(ax.xaxis.get_ticks_position())
    if(yLabel != None):
        ax.set_ylabel(yLabel, labelpad=2)
        ax.yaxis.set_label_position(ax.yaxis.get_ticks_position())


def sum1(x, yL1, yH1, yL2, yH2):
    return (yH2+x[3])*(yL1-x[0])-(yL2-x[2])*(yH1+x[1])


def sum2(x, yL1, yH1, yL2, yH2):
    return abs((x[0]+x[1])/(yH1-yL1)-(x[2]+x[3])/(yH2-yL2))+(x[0]+x[1])/(yH1-yL1)+(x[2]+x[3])/(yH2-yL2)


def align0(ax1, ax2):
    """
    Align the y axis of 2 different axist object such that the 0 is at the same height.
    Assumption: the yLow and yHigh is <0 and >0 respectively. If it is not, then 0 will be assigned
    >>> align0(ax1=ax1,ax2=ax2)
    """
    yL1, yH1 = ax1.get_ylim()
    yL1 = min(0, yL1)
    yH1 = max(0, yH1)
    setAx(ax1, yLow=yL1, yHigh=yH1)
    yL2, yH2 = ax2.get_ylim()
    yL2 = min(0, yL2)
    yH2 = max(0, yH2)
    setAx(ax2, yLow=yL2, yHigh=yH2)
    yList = [yL1, yH1, yL2, yH2]
    delta = np.array([0, 0, 0, 0])
    # res = minimize(x0=delta, fun=(lambda x1: x1.sum()), constraints=({'type': 'eq', 'fun': (
    #     lambda x1: sum1(x1, *yList))}), bounds=((0, None), (0, None), (0, None), (0, None)))
    res = minimize(x0=delta, fun=(lambda x1: sum2(x1, *yList)), constraints=({'type': 'eq', 'fun': (
        lambda x1: sum1(x1, *yList))}), bounds=((0, None), (0, None), (0, None), (0, None)))
    delta = res.x
    setAx(ax=ax1, yLow=yL1-delta[0], yHigh=yH1+delta[1])
    setAx(ax=ax2, yLow=yL2-delta[2], yHigh=yH2+delta[3])
    return res.x

# %% [markdown]
# ### Plot data
# The functions below are used to plot data including adding legend given that canvas and data are suitable

# %%


def plot1Graph(dataList, typeList, labelList, axObj, isPlot=True, markerEdgeWidth=3):
    """
    Given a list of data, the plot type, and axis object. Plot the data and return the axis object
    Argument:
        dataList = list of dataframe to be plotted. The size has to match the type.
        typeList = list of formatstring. len(typeList) >= len(dataList), else type is repeated.
                   exception --> if the ;repeat keyword are used.
                   example of accepted input:
                   ['ro','ks-'], ['auto0;o;auto2','auto0;auto0;-',';;auto0'],['r;auto0;-;repeat'] 
        labelList = list of label. len(labelList) >= len(dataList), else label is repeated.
        axObj = axis object
    Optional argument:
        isPlot -> if true, the graph will be shown
    Limitation:
        dataList can only consist 2D dataframe
    Default setting:
        fillstyle = none
        linewidth = 2
    """
    dataSize = len(dataList)
    typeSize = len(typeList)
    labelSize = len(labelList)
    if('repeat' in typeList[0]):
        # extract the first type
        typeFirst = typeList[0].split(';')
        colorType = typeFirst[0]
        markerType = typeFirst[1]
        lineType = typeFirst[2]
        colorNum = -1
        markerNum = -1
        lineNum = -1
        if('auto' in colorType):
            colorNum = int(colorType[4:len(colorType)])
        if('auto' in markerType):
            markerNum = int(markerType[4:len(markerType)])
        if('auto' in lineType):
            lineNum = int(lineType[4:len(lineType)])
        typeList = ['']*dataSize
        for ind in range(dataSize):
            currType = ''
            if(colorNum > -1):
                currType = currType+'auto'+str(colorNum)+';'
                colorNum = colorNum+1
            else:
                currType = currType+colorType+';'
            if(markerNum > -1):
                currType = currType+'auto'+str(markerNum)+';'
                markerNum = markerNum+1
            else:
                currType = currType+markerType+';'
            if(lineNum > -1):
                currType = currType+'auto'+str(lineNum)
                lineNum = lineNum+1
            else:
                currType = currType+lineType
            typeList[ind] = currType
        typeSize = dataSize
    for ind in range(dataSize):
        data1 = dataList[ind]
        currType = autoType(typeList[ind % typeSize])
        axObj.plot(data1.iloc[:, 0], data1.iloc[:, 1], currType, fillstyle='none', linewidth=2,
                   markersize=8, markeredgewidth=markerEdgeWidth, label=labelList[ind % labelSize])
        setAx(ax=axObj, xLabel=data1.columns[0], yLabel=data1.columns[1])
    if(isPlot):
        axObj.figure.show()
    return axObj


def createLegend(axPrimer, axList=None, boxLoc=[1, 1, 1]):
    """
    Combining the label in axis list into a legend and draw it wrt axPrimer
    >>> createLegend(axPrimer,axList=[ax1,ax2],boxLoc=[1,1,1])
    >>> createLegend(axPrimer)
    """
    if(axList == None):
        axList = [axPrimer]
    if(len(boxLoc) < 3):
        boxLoc = [1, 1, 1]
    hList = []
    lList = []
    for ax in axList:
        h, l = ax.get_legend_handles_labels()
        hList = hList+h
        lList = lList+l
    axPrimer.legend(handles=hList, labels=lList, bbox_to_anchor=(
        boxLoc[0], boxLoc[1]), loc=boxLoc[2], frameon=True, fontsize=16)
    return axPrimer


def autoType(typeVal):
    """
    create the marker type
    example of accepted format for typeVal: 'ro-', 'r;o;-', 'auto0;auto1;auto0','auto0;o;-'
    Note if 'auto' is used the format must be '<colorType>;<markerType>;<lineType>'. Each of this can be auto type.
    """
    colorList = ['k', 'r', 'b', 'g', 'm']
    colorLen = len(colorList)
    markerList = ['o', 's', '^', 'x', 'D']
    markerLen = len(markerList)
    lineList = ['-', '--', ':', '-.']
    lineLen = len(lineList)
    currType = ''
    typeArr = typeVal.split(';')
    if(len(typeArr) < 3):
        return typeVal
    colorType = typeArr[0]
    markerType = typeArr[1]
    lineType = typeArr[2]
    colorNum = -1
    markerNum = -1
    lineNum = -1
    if('auto' in colorType):
        colorNum = int(colorType[4:len(colorType)])
    if('auto' in markerType):
        markerNum = int(markerType[4:len(markerType)])
    if('auto' in lineType):
        lineNum = int(lineType[4:len(lineType)])
    if(colorNum > -1):
        currType = currType+colorList[colorNum % colorLen]
    else:
        currType = currType + colorType
        colorLen = 1
    if(markerNum > -1):
        currType = currType+markerList[(markerNum//colorLen) % markerLen]
    else:
        currType = currType + markerType
        markerLen = 1
    if(lineNum > -1):
        currType = currType+lineList[(lineNum//(colorLen*markerLen)) % lineLen]
    else:
        currType = currType + lineType
    return currType
