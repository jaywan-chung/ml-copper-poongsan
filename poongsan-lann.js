// Construct Problem-specific Aritifical Neural Networks.
// This is for prediction of strength and conductivity of copper alloys.
// Programmed by Dr. Jaywan Chung
// v0.1 updated on Nov 15, 2023

"use strict";

const jcApp = {
    chartHeight: 500,
    minTime: 0,
    maxTime: 1000,
    nTimeNodes: 101,
    dataLegend: 'Expt',
    plot1Legend: 'Pred 1',
    plot2Legend: 'Pred 2',
    colorRawData: '#594D5B',
    colorPlot1: '#808080',  // gray
    colorPlot2: '#1976D2',  // blue
};

class CopperLann {
    constructor(embeddingNet, f1Net, f2Net) {
        this.embeddingNet = embeddingNet;
        this.f1Net = f1Net;
        this.f2Net = f2Net;
        this.embeddingInput = new Matrix(9, 1);
        this.dictionaryVar = new Matrix(8, 1);
        this.latentVar = null;
        this.meanOutputMatrix = new Matrix(2, 1);
        this.stdOutputMatrix = new Matrix(2, 1);
    }
    evaluate(inputMatrix) {
        const scaledInput = CopperLann.getScaledInput(inputMatrix);
        for (let i=0; i<9; i++) {
            this.embeddingInput.setElement(i, 0, scaledInput.getElement(i, 0));
        }
        let scaledTime = scaledInput.getElement(10, 0);
        this.embeddingNet.evaluate(this.embeddingInput);
        this.latentVar = this.embeddingNet.outputMatrix;
        for (let i=0; i<6; i++) {
            this.dictionaryVar.setElement(i, 0, this.latentVar.getElement(i, 0));
        }
        for (let i=6; i<8; i++) {
            this.dictionaryVar.setElement(i, 0, scaledInput.getElement(i+3, 0));
        }
        this.f1Net.evaluate(this.dictionaryVar);
        this.f2Net.evaluate(this.latentVar);
        let f1Output = this.f1Net.outputMatrix;
        let f2Output = this.f2Net.outputMatrix;
        // Update the mean and std
        for (let i=0; i<2; i++) {
            this.meanOutputMatrix.setElement(i, 0, scaledTime * f1Output.getElement(i, 0) + f2Output.getElement(i, 0));
            this.stdOutputMatrix.setElement(i, 0, scaledTime * f1Output.getElement(i+2, 0) + f2Output.getElement(i+2, 0));
        }
        this.scaleOutput();
    }
    static getScaledInput(inputMatrix) {
        const scaledInput = inputMatrix.clone();
        // Input: Ni, Co, Si, Cr, Mn, SHT.0, SHT.1, 1st Aging temp, 1st Aging time, Aging temp, Aging time
        scaledInput.array[0] /= 2.5;  // Ni
        scaledInput.array[1] /= 0.5;  // Co
        scaledInput.array[2] /= 0.5;  // Si
        scaledInput.array[3] /= 0.1;  // Cr
        scaledInput.array[4] /= 0.1;  // Mn
        scaledInput.array[5] /= 1.0;  // SHT.0
        scaledInput.array[6] /= 1.0;  // SHT.1
        scaledInput.array[7] /= 500.0;  // 1st Aging temp
        scaledInput.array[8] = Math.log(1.0 + scaledInput.array[8]);  // 1st Aging time
        scaledInput.array[9] /= 500.0;  // Aging temp
        scaledInput.array[10] = Math.log(1.0 + scaledInput.array[10]);  // Aging time

        return scaledInput;
    }
    scaleOutput() {
        for (let i=0; i<2; i++) {
            let mean = this.meanOutputMatrix.getElement(i, 0);
            let std = this.stdOutputMatrix.getElement(i, 0);
            this.meanOutputMatrix.setElement(i, 0, Math.log(Math.exp(mean) + 1.0));  // softplus activation
            this.stdOutputMatrix.setElement(i, 0, Math.log(Math.exp(std) + 1.0));
        }
        this.meanOutputMatrix.array[0] *= 40.0   // electrical conductivity [%IACS]
        this.meanOutputMatrix.array[1] *= 200.0;  // Vickers hardness [HV]
        this.stdOutputMatrix.array[0] *= 40.0;
        this.stdOutputMatrix.array[1] *= 200.0;
    }
}

class CopperFcnn {
    constructor(fcnn) {
        this.fcnn = fcnn;
        this.meanOutputMatrix = new Matrix(2, 1);
        this.stdOutputMatrix = new Matrix(2, 1);
    }
    evaluate(inputMatrix) {
        const scaledInput = CopperFcnn.getScaledInput(inputMatrix);
        this.fcnn.evaluate(scaledInput);
        let output = this.fcnn.outputMatrix;
        // Update the mean and std
        for (let i=0; i<2; i++) {
            this.meanOutputMatrix.setElement(i, 0, output.getElement(i, 0));
            this.stdOutputMatrix.setElement(i, 0, output.getElement(i+2, 0));
        }
        this.scaleOutput();
    }
    static getScaledInput(inputMatrix) {
        const scaledInput = inputMatrix.clone();
        // Input: Ni, Co, Si, Cr, Mn, SHT.0, SHT.1, 1st Aging temp, 1st Aging time, Aging temp, Aging time
        scaledInput.array[0] /= 2.5;  // Ni
        scaledInput.array[1] /= 0.5;  // Co
        scaledInput.array[2] /= 0.5;  // Si
        scaledInput.array[3] /= 0.1;  // Cr
        scaledInput.array[4] /= 0.1;  // Mn
        scaledInput.array[5] /= 1.0;  // SHT.0
        scaledInput.array[6] /= 1.0;  // SHT.1
        scaledInput.array[7] /= 500.0;  // 1st Aging temp
        scaledInput.array[8] = Math.log(1.0 + scaledInput.array[8]);  // 1st Aging time
        scaledInput.array[9] /= 500.0;  // Aging temp
        scaledInput.array[10] = Math.log(1.0 + scaledInput.array[10]);  // Aging time

        return scaledInput;
    }
    scaleOutput() {
        this.meanOutputMatrix.array[0] *= 40.0   // electrical conductivity [%IACS]
        this.meanOutputMatrix.array[1] *= 200.0;  // Vickers hardness [HV]
        this.stdOutputMatrix.array[0] *= 40.0;
        this.stdOutputMatrix.array[1] *= 200.0;
    }
}

jcApp.startApp = function() {
    console.log("Starting App...");
    jcApp.initSelectRawdata();
    jcApp.initLann();
    jcApp.initFcnn();

    jcApp.timeArray = jcApp.getLinearSpace(jcApp.minTime, jcApp.maxTime, jcApp.nTimeNodes);
    jcApp.plot1Input = new Matrix(11, 1);  // Ni, Co, Si, Cr, Mn, SHT.0, SHT.1, 1st Aging temp, 1st Aging time, Aging temp, Aging time
    jcApp.plot1HardnessArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot1ConductivityArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot1HardnessStdArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot1ConductivityStdArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot1FcnnHardnessArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot1FcnnConductivityArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot1FcnnHardnessStdArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot1FcnnConductivityStdArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot2Input = new Matrix(11, 1);
    jcApp.plot2HardnessArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot2ConductivityArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot2HardnessStdArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot2ConductivityStdArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot2FcnnHardnessArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot2FcnnConductivityArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot2FcnnHardnessStdArray = new Float64Array(jcApp.nTimeNodes);
    jcApp.plot2FcnnConductivityStdArray = new Float64Array(jcApp.nTimeNodes);
    console.log('Memory allocated.');

    google.charts.load('current', {'packages':['corechart']});
    google.charts.setOnLoadCallback(jcApp.activateChartsAndButtons); // activate buttons when google charts is loaded.
}

jcApp.initSelectRawdata = function() {
    jcApp.select = document.getElementById("select-rawdata");
    for (const key of Object.keys(jcApp.rawdata)) {
        let opt = document.createElement("option");
        opt.value = key;
        opt.innerHTML = key;
        jcApp.select.appendChild(opt);
    }
    const selectButtonForPlot1 = document.getElementById("select-rawdata-button-for-plot1");
    selectButtonForPlot1.addEventListener("click", jcApp.onClickSelectDataButtonForPlot1);
    const selectButtonForPlot2 = document.getElementById("select-rawdata-button-for-plot2");
    selectButtonForPlot2.addEventListener("click", jcApp.onClickSelectDataButtonForPlot2);
    // select the first data
    // jcApp.select.options[0].selected = true;
    // jcApp.select.value = "Ni1.77, Co0.31, Si0.45, Cr0.13, Mn0.13, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC";
    jcApp.select.value = "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, aging=460 degC";
    jcApp.onClickSelectDataButtonForPlot1();
    console.log("'Select Data' initialized.");
}

jcApp.onClickSelectDataButtonForPlot1 = function() {
    let dataName = jcApp.select.value;
    if (!dataName) return;  // if not selected, do nothing.
    let input = jcApp.rawdata[dataName]["input"];
    document.getElementById("plot1-Ni-composition").value = input[0];
    document.getElementById("plot1-Co-composition").value = input[1];
    document.getElementById("plot1-Si-composition").value = input[2];
    document.getElementById("plot1-Cr-composition").value = input[3];
    document.getElementById("plot1-Mn-composition").value = input[4];
    if (input[5]) {  // hand input[5] and input[6]
        document.getElementById("plot1-SHT").value = "Pilot";
    } else {
        document.getElementById("plot1-SHT").value = "AP7";
    }
    document.getElementById("plot1-1st-aging-temp").value = input[7];
    document.getElementById("plot1-1st-aging-time").value = input[8];
    document.getElementById("plot1-aging-temp").value = input[9];
}

jcApp.onClickSelectDataButtonForPlot2 = function() {
    let dataName = jcApp.select.value;
    if (!dataName) return;  // if not selected, do nothing.
    let input = jcApp.rawdata[dataName]["input"];
    document.getElementById("plot2-Ni-composition").value = input[0];
    document.getElementById("plot2-Co-composition").value = input[1];
    document.getElementById("plot2-Si-composition").value = input[2];
    document.getElementById("plot2-Cr-composition").value = input[3];
    document.getElementById("plot2-Mn-composition").value = input[4];
    if (input[5]) {  // hand input[5] and input[6]
        document.getElementById("plot2-SHT").value = "Pilot";
    } else {
        document.getElementById("plot2-SHT").value = "AP7";
    }
    document.getElementById("plot2-1st-aging-temp").value = input[7];
    document.getElementById("plot2-1st-aging-time").value = input[8];
    document.getElementById("plot2-aging-temp").value = input[9];
}

jcApp.activateChartsAndButtons = function() {
    jcApp.initMatPropCharts();

    document.getElementById("predict-mat-prop").addEventListener("click", function() {
        if (jcApp.predict()) {   // draw chart only when the prediction is successful.
            jcApp.drawCharts();
        } else {
            console.log("Prediction failed.");
        }
    });
}

jcApp.predict = function() {
    jcApp.clearPrediction();

    const plot1NiComposition = parseFloat(document.getElementById("plot1-Ni-composition").value);
    const plot1CoComposition = parseFloat(document.getElementById("plot1-Co-composition").value);
    const plot1SiComposition = parseFloat(document.getElementById("plot1-Si-composition").value);
    const plot1CrComposition = parseFloat(document.getElementById("plot1-Cr-composition").value);
    const plot1MnComposition = parseFloat(document.getElementById("plot1-Mn-composition").value);
    const plot1SHT = document.getElementById("plot1-SHT").value;
    const plot1FirstAgingTemp = parseFloat(document.getElementById("plot1-1st-aging-temp").value);
    const plot1FirstAgingTime = parseFloat(document.getElementById("plot1-1st-aging-time").value);
    const plot1AgingTemp = parseFloat(document.getElementById("plot1-aging-temp").value);

    const plot2NiComposition = parseFloat(document.getElementById("plot2-Ni-composition").value);
    const plot2CoComposition = parseFloat(document.getElementById("plot2-Co-composition").value);
    const plot2SiComposition = parseFloat(document.getElementById("plot2-Si-composition").value);
    const plot2CrComposition = parseFloat(document.getElementById("plot2-Cr-composition").value);
    const plot2MnComposition = parseFloat(document.getElementById("plot2-Mn-composition").value);
    const plot2SHT = document.getElementById("plot2-SHT").value;
    const plot2FirstAgingTemp = parseFloat(document.getElementById("plot2-1st-aging-temp").value);
    const plot2FirstAgingTime = parseFloat(document.getElementById("plot2-1st-aging-time").value);
    const plot2AgingTemp = parseFloat(document.getElementById("plot2-aging-temp").value);

    // check the validity of numbers
    if (!(Number.isFinite(plot1NiComposition) && Number.isFinite(plot1CoComposition) && Number.isFinite(plot1SiComposition) && Number.isFinite(plot1CrComposition) && Number.isFinite(plot1MnComposition))) {
        window.alert("Composition in Pred 1 is not valid!");
        return false;
    }
    if (!(Number.isFinite(plot1FirstAgingTemp) && Number.isFinite(plot1FirstAgingTime) && Number.isFinite(plot1AgingTemp))) {
        window.alert("Aging in Pred 1 is not valid!");
        return false;
    }
    if (!(Number.isFinite(plot2NiComposition) && Number.isFinite(plot2CoComposition) && Number.isFinite(plot2SiComposition) && Number.isFinite(plot2CrComposition) && Number.isFinite(plot2MnComposition))) {
        window.alert("Composition in Pred 2 is not valid!");
        return false;
    }
    if (!(Number.isFinite(plot2FirstAgingTemp) && Number.isFinite(plot2FirstAgingTime) && Number.isFinite(plot2AgingTemp))) {
        window.alert("Aging in Pred 2 is not valid!");
        return false;
    }
    // check the non-negativity of aging info
    if ((plot1FirstAgingTemp < 0) || (plot1FirstAgingTime < 0) || (plot1AgingTemp < 0)) {
        window.alert("Aging params in Pred 1 must be non-negative!");
        return false;
    }
    if ((plot2FirstAgingTemp < 0) || (plot2FirstAgingTime < 0) || (plot2AgingTemp < 0)) {
        window.alert("Aging params in Pred 2 must be non-negative!");
        return false;
    }

    // make prediction for Pred 1
    jcApp.plot1Input.setElement(0, 0, plot1NiComposition);
    jcApp.plot1Input.setElement(1, 0, plot1CoComposition);
    jcApp.plot1Input.setElement(2, 0, plot1SiComposition);
    jcApp.plot1Input.setElement(3, 0, plot1CrComposition);
    jcApp.plot1Input.setElement(4, 0, plot1MnComposition);
    if (plot1SHT == 'Pilot') {
        jcApp.plot1Input.setElement(5, 0, 1);
        jcApp.plot1Input.setElement(6, 0, 0);
    } else {
        jcApp.plot1Input.setElement(5, 0, 0);
        jcApp.plot1Input.setElement(6, 0, 1);
    }
    jcApp.plot1Input.setElement(7, 0, plot1FirstAgingTemp);
    jcApp.plot1Input.setElement(8, 0, plot1FirstAgingTime);
    jcApp.plot1Input.setElement(9, 0, plot1AgingTemp);
    for(let i=0; i<jcApp.nTimeNodes; i++) {
        jcApp.plot1Input.setElement(10, 0, jcApp.timeArray[i]);
        // prediction by LaNN+TL
        jcApp.lann.evaluate(jcApp.plot1Input);
        jcApp.plot1ConductivityArray[i] = jcApp.lann.meanOutputMatrix.array[0];
        jcApp.plot1HardnessArray[i] = jcApp.lann.meanOutputMatrix.array[1];
        jcApp.plot1ConductivityStdArray[i] = jcApp.lann.stdOutputMatrix.array[0];
        jcApp.plot1HardnessStdArray[i] = jcApp.lann.stdOutputMatrix.array[1];
        // prediction by FCNN
        jcApp.fcnn.evaluate(jcApp.plot1Input);
        jcApp.plot1FcnnConductivityArray[i] = jcApp.fcnn.meanOutputMatrix.array[0];
        jcApp.plot1FcnnHardnessArray[i] = jcApp.fcnn.meanOutputMatrix.array[1];
        jcApp.plot1FcnnConductivityStdArray[i] = jcApp.fcnn.stdOutputMatrix.array[0];
        jcApp.plot1FcnnHardnessStdArray[i] = jcApp.fcnn.stdOutputMatrix.array[1];
    }

    // make prediction for Pred 2
    jcApp.plot2Input.setElement(0, 0, plot2NiComposition);
    jcApp.plot2Input.setElement(1, 0, plot2CoComposition);
    jcApp.plot2Input.setElement(2, 0, plot2SiComposition);
    jcApp.plot2Input.setElement(3, 0, plot2CrComposition);
    jcApp.plot2Input.setElement(4, 0, plot2MnComposition);
    if (plot2SHT == 'Pilot') {
        jcApp.plot2Input.setElement(5, 0, 1);
        jcApp.plot2Input.setElement(6, 0, 0);
    } else {
        jcApp.plot2Input.setElement(5, 0, 0);
        jcApp.plot2Input.setElement(6, 0, 1);
    }
    jcApp.plot2Input.setElement(7, 0, plot2FirstAgingTemp);
    jcApp.plot2Input.setElement(8, 0, plot2FirstAgingTime);
    jcApp.plot2Input.setElement(9, 0, plot2AgingTemp);
    for(let i=0; i<jcApp.nTimeNodes; i++) {
        jcApp.plot2Input.setElement(10, 0, jcApp.timeArray[i]);
        // prediction by LaNN+TL
        jcApp.lann.evaluate(jcApp.plot2Input);
        jcApp.plot2ConductivityArray[i] = jcApp.lann.meanOutputMatrix.array[0];
        jcApp.plot2HardnessArray[i] = jcApp.lann.meanOutputMatrix.array[1];
        jcApp.plot2ConductivityStdArray[i] = jcApp.lann.stdOutputMatrix.array[0];
        jcApp.plot2HardnessStdArray[i] = jcApp.lann.stdOutputMatrix.array[1];
        // prediction by FCNN
        jcApp.fcnn.evaluate(jcApp.plot2Input);
        jcApp.plot2FcnnConductivityArray[i] = jcApp.fcnn.meanOutputMatrix.array[0];
        jcApp.plot2FcnnHardnessArray[i] = jcApp.fcnn.meanOutputMatrix.array[1];
        jcApp.plot2FcnnConductivityStdArray[i] = jcApp.fcnn.stdOutputMatrix.array[0];
        jcApp.plot2FcnnHardnessStdArray[i] = jcApp.fcnn.stdOutputMatrix.array[1];
    }

    console.log("Prediction complete.");

    return true;
}

jcApp.clearPrediction = function() {
    jcApp.plot1Input.fill(NaN);
    jcApp.plot1HardnessArray.fill(NaN);
    jcApp.plot1ConductivityArray.fill(NaN);
    jcApp.plot1FcnnHardnessArray.fill(NaN);
    jcApp.plot1FcnnConductivityArray.fill(NaN);
    jcApp.plot2Input.fill(NaN);
    jcApp.plot2HardnessArray.fill(NaN);
    jcApp.plot2ConductivityArray.fill(NaN);
    jcApp.plot2FcnnHardnessArray.fill(NaN);
    jcApp.plot2FcnnConductivityArray.fill(NaN);

    console.log("Prediction cleared.");
}

jcApp.checkShowOptions = function() {
    jcApp.showData = document.getElementById("show-data").checked;
    jcApp.showPlot1 = document.getElementById("show-plot1").checked;
    jcApp.showPlot1Ci = document.getElementById("show-plot1-ci").checked;
    jcApp.showPlot2 = document.getElementById("show-plot2").checked;
    jcApp.showPlot2Ci = document.getElementById("show-plot2-ci").checked;
}

jcApp.initMatPropCharts = function() {
    jcApp.chartHardness = new google.visualization.ComboChart(document.getElementById('chart-hardness'));
    jcApp.chartConductivity = new google.visualization.ComboChart(document.getElementById('chart-conductivity'));
    jcApp.chartFcnnHardness = new google.visualization.ComboChart(document.getElementById('chart-fcnn-hardness'));
    jcApp.chartFcnnConductivity = new google.visualization.ComboChart(document.getElementById('chart-fcnn-conductivity'));
    console.log("Charts initialized.");
}

jcApp.drawCharts = function() {
    jcApp.checkShowOptions();
    jcApp.drawHardnessChart();
    jcApp.drawConductivityChart();
    jcApp.drawFcnnHardnessChart();
    jcApp.drawFcnnConductivityChart();
}

jcApp.drawHardnessChart = function() {
    const chart = jcApp.chartHardness;
    const xLabel = "Aging time (min)";
    const yLabel = "Vickers Hardness (HV)";
    const yScale = 1.0;   // [HV]
    const selectedDataName = jcApp.select.value;

    let data = new google.visualization.DataTable();
    data.addColumn('number', xLabel); 
    data.addColumn('number', jcApp.dataLegend);
    data.addColumn('number', jcApp.plot1Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn('number', jcApp.plot2Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});

    if(selectedDataName && jcApp.showData) {
        let timeData = jcApp.rawdata[selectedDataName]["time [min]"];
        let propData = jcApp.rawdata[selectedDataName]["hardness [HV]"];
        for(let i=0; i<timeData.length; i++) {
            data.addRow([timeData[i], propData[i]*yScale, NaN, NaN, NaN, NaN, NaN, NaN]);
        }    
    }
    let plot1Mean, plot1Std, plot2Mean, plot2Std;
    for(let i=0; i<jcApp.nTimeNodes; i++) {
        plot1Mean = plot1Std = plot2Mean = plot2Std = NaN;
        if(jcApp.showPlot1) {
            plot1Mean = jcApp.plot1HardnessArray[i]*yScale;
        }
        if(jcApp.showPlot1Ci) {
            plot1Std = jcApp.plot1HardnessStdArray[i]*yScale;
        }
        if(jcApp.showPlot2) {
            plot2Mean = jcApp.plot2HardnessArray[i]*yScale;
        }
        if(jcApp.showPlot2Ci) {
            plot2Std = jcApp.plot2HardnessStdArray[i]*yScale;
        }
        data.addRow([jcApp.timeArray[i], NaN,
            plot1Mean, plot1Mean-1.96*plot1Std, plot1Mean+1.96*plot1Std,
            plot2Mean, plot2Mean-1.96*plot2Std, plot2Mean+1.96*plot2Std]);
    }

    let options = {
      seriesType: 'line',
      series: {0: {type: 'scatter'}},
      title: yLabel,
      titleTextStyle: {bold: true, fontSize: 20,},
      hAxis: {title: xLabel, titleTextStyle: {italic: false, fontSize: 15,}, viewWindow: {min: -1.0}},
      vAxis: {title: yLabel, titleTextStyle: {italic: false, fontSize: 15,},},
      legend: { position: 'bottom', alignment: 'center' },
      intervals: { style: 'area' },
      colors: [jcApp.colorRawData, jcApp.colorPlot1, jcApp.colorPlot2],
      height: jcApp.chartHeight,
    };
  
    chart.draw(data, options);
};

jcApp.drawConductivityChart = function() {
    const chart = jcApp.chartConductivity;
    const xLabel = "Aging time (min)";
    const yLabel = "Conductivity (%IACS)";
    const yScale = 1.0;   // [%IACS]
    const selectedDataName = jcApp.select.value;

    let data = new google.visualization.DataTable();
    data.addColumn('number', xLabel); 
    data.addColumn('number', jcApp.dataLegend);
    data.addColumn('number', jcApp.plot1Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn('number', jcApp.plot2Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});

    if(selectedDataName && jcApp.showData) {
        let timeData = jcApp.rawdata[selectedDataName]["time [min]"];
        let propData = jcApp.rawdata[selectedDataName]["conductivity [%IACS]"];
        for(let i=0; i<timeData.length; i++) {
            data.addRow([timeData[i], propData[i]*yScale, NaN, NaN, NaN, NaN, NaN, NaN]);
        }    
    }
    let plot1Mean, plot1Std, plot2Mean, plot2Std;
    for(let i=0; i<jcApp.nTimeNodes; i++) {
        plot1Mean = plot1Std = plot2Mean = plot2Std = NaN;
        if(jcApp.showPlot1) {
            plot1Mean = jcApp.plot1ConductivityArray[i]*yScale;
        }
        if(jcApp.showPlot1Ci) {
            plot1Std = jcApp.plot1ConductivityStdArray[i]*yScale;
        }
        if(jcApp.showPlot2) {
            plot2Mean = jcApp.plot2ConductivityArray[i]*yScale;
        }
        if(jcApp.showPlot2Ci) {
            plot2Std = jcApp.plot2ConductivityStdArray[i]*yScale;
        }
        data.addRow([jcApp.timeArray[i], NaN,
            plot1Mean, plot1Mean-1.96*plot1Std, plot1Mean+1.96*plot1Std,
            plot2Mean, plot2Mean-1.96*plot2Std, plot2Mean+1.96*plot2Std]);
    }

    let options = {
      seriesType: 'line',
      series: {0: {type: 'scatter'}},
      title: yLabel,
      titleTextStyle: {bold: true, fontSize: 20,},
      hAxis: {title: xLabel, titleTextStyle: {italic: false, fontSize: 15,}, viewWindow: {min: -1.0}},
      vAxis: {title: yLabel, titleTextStyle: {italic: false, fontSize: 15,},},
      legend: { position: 'bottom', alignment: 'center' },
      intervals: { style: 'area' },
      colors: [jcApp.colorRawData, jcApp.colorPlot1, jcApp.colorPlot2],
      height: jcApp.chartHeight,
    };
  
    chart.draw(data, options);
};

jcApp.drawFcnnHardnessChart = function() {
    const chart = jcApp.chartFcnnHardness;
    const xLabel = "Aging time (min)";
    const yLabel = "Vickers Hardness (HV)";
    const yScale = 1.0;   // [HV]
    const selectedDataName = jcApp.select.value;

    let data = new google.visualization.DataTable();
    data.addColumn('number', xLabel); 
    data.addColumn('number', jcApp.dataLegend);
    data.addColumn('number', jcApp.plot1Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn('number', jcApp.plot2Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});

    if(selectedDataName && jcApp.showData) {
        let timeData = jcApp.rawdata[selectedDataName]["time [min]"];
        let propData = jcApp.rawdata[selectedDataName]["hardness [HV]"];
        for(let i=0; i<timeData.length; i++) {
            data.addRow([timeData[i], propData[i]*yScale, NaN, NaN, NaN, NaN, NaN, NaN]);
        }    
    }
    let plot1Mean, plot1Std, plot2Mean, plot2Std;
    for(let i=0; i<jcApp.nTimeNodes; i++) {
        plot1Mean = plot1Std = plot2Mean = plot2Std = NaN;
        if(jcApp.showPlot1) {
            plot1Mean = jcApp.plot1FcnnHardnessArray[i]*yScale;
        }
        if(jcApp.showPlot1Ci) {
            plot1Std = jcApp.plot1FcnnHardnessStdArray[i]*yScale;
        }
        if(jcApp.showPlot2) {
            plot2Mean = jcApp.plot2FcnnHardnessArray[i]*yScale;
        }
        if(jcApp.showPlot2Ci) {
            plot2Std = jcApp.plot2FcnnHardnessStdArray[i]*yScale;
        }
        data.addRow([jcApp.timeArray[i], NaN,
            plot1Mean, plot1Mean-1.96*plot1Std, plot1Mean+1.96*plot1Std,
            plot2Mean, plot2Mean-1.96*plot2Std, plot2Mean+1.96*plot2Std]);
    }

    let options = {
      seriesType: 'line',
      series: {0: {type: 'scatter'}},
      title: yLabel,
      titleTextStyle: {bold: true, fontSize: 20,},
      hAxis: {title: xLabel, titleTextStyle: {italic: false, fontSize: 15,}, viewWindow: {min: -1.0}},
      vAxis: {title: yLabel, titleTextStyle: {italic: false, fontSize: 15,},},
      legend: { position: 'bottom', alignment: 'center' },
      intervals: { style: 'area' },
      colors: [jcApp.colorRawData, jcApp.colorPlot1, jcApp.colorPlot2],
      height: jcApp.chartHeight,
    };
  
    chart.draw(data, options);
};

jcApp.drawFcnnConductivityChart = function() {
    const chart = jcApp.chartFcnnConductivity;
    const xLabel = "Aging time (min)";
    const yLabel = "Conductivity (%IACS)";
    const yScale = 1.0;   // [%IACS]
    const selectedDataName = jcApp.select.value;

    let data = new google.visualization.DataTable();
    data.addColumn('number', xLabel); 
    data.addColumn('number', jcApp.dataLegend);
    data.addColumn('number', jcApp.plot1Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn('number', jcApp.plot2Legend);
    data.addColumn({type: 'number', role: 'interval'});
    data.addColumn({type: 'number', role: 'interval'});

    if(selectedDataName && jcApp.showData) {
        let timeData = jcApp.rawdata[selectedDataName]["time [min]"];
        let propData = jcApp.rawdata[selectedDataName]["conductivity [%IACS]"];
        for(let i=0; i<timeData.length; i++) {
            data.addRow([timeData[i], propData[i]*yScale, NaN, NaN, NaN, NaN, NaN, NaN]);
        }    
    }
    let plot1Mean, plot1Std, plot2Mean, plot2Std;
    for(let i=0; i<jcApp.nTimeNodes; i++) {
        plot1Mean = plot1Std = plot2Mean = plot2Std = NaN;
        if(jcApp.showPlot1) {
            plot1Mean = jcApp.plot1FcnnConductivityArray[i]*yScale;
        }
        if(jcApp.showPlot1Ci) {
            plot1Std = jcApp.plot1FcnnConductivityStdArray[i]*yScale;
        }
        if(jcApp.showPlot2) {
            plot2Mean = jcApp.plot2FcnnConductivityArray[i]*yScale;
        }
        if(jcApp.showPlot2Ci) {
            plot2Std = jcApp.plot2FcnnConductivityStdArray[i]*yScale;
        }
        data.addRow([jcApp.timeArray[i], NaN,
            plot1Mean, plot1Mean-1.96*plot1Std, plot1Mean+1.96*plot1Std,
            plot2Mean, plot2Mean-1.96*plot2Std, plot2Mean+1.96*plot2Std]);
    }

    let options = {
      seriesType: 'line',
      series: {0: {type: 'scatter'}},
      title: yLabel,
      titleTextStyle: {bold: true, fontSize: 20,},
      hAxis: {title: xLabel, titleTextStyle: {italic: false, fontSize: 15,}, viewWindow: {min: -1.0}},
      vAxis: {title: yLabel, titleTextStyle: {italic: false, fontSize: 15,},},
      legend: { position: 'bottom', alignment: 'center' },
      intervals: { style: 'area' },
      colors: [jcApp.colorRawData, jcApp.colorPlot1, jcApp.colorPlot2],
      height: jcApp.chartHeight,
    };
  
    chart.draw(data, options);
};


jcApp.initLann = function() {
    let jsonObj = jcApp.jsonObjPoongsanLann;
    let embeddingNet = new FullyConnectedNeuralNetwork(9,  // 11 inputs - 2 surface vars
        jsonObj["embeddingNet"]["weightsArray"],
        jsonObj["embeddingNet"]["biasesArray"],
        jsonObj["embeddingNet"]["activationArray"]
    );
     let f1Net = new FullyConnectedNeuralNetwork(8,  // 
        jsonObj["f1Net"]["weightsArray"],
        jsonObj["f1Net"]["biasesArray"],
        jsonObj["f1Net"]["activationArray"]
    );
    let f2Net = new FullyConnectedNeuralNetwork(6,
        jsonObj["f2Net"]["weightsArray"],
        jsonObj["f2Net"]["biasesArray"],
        jsonObj["f2Net"]["activationArray"]
    );
    jcApp.lann = new CopperLann(embeddingNet, f1Net, f2Net);
    console.log("Machine learning model (LaNN) initialized.");
};

jcApp.initFcnn = function() {
    let jsonObj = jcApp.jsonObjPoongsanFcnn;
    let fcnn = new FullyConnectedNeuralNetwork(11,  // 11 inputs
        jsonObj["FCNN"]["weightsArray"],
        jsonObj["FCNN"]["biasesArray"],
        jsonObj["FCNN"]["activationArray"]
    );
    jcApp.fcnn = new CopperFcnn(fcnn);
    console.log("Machine learning model (FCNN) initialized.");
};

jcApp.getLinearSpace = function(x0, xf, numNodes) {
    const vec = new Float64Array(numNodes);
    const dx = (xf-x0)/(numNodes-1);
    for(let i=0; i<vec.length; i++) {
        vec[i] = (x0 + dx*i);
    };
    vec[vec.length-1] = xf;

    return vec;
};

// Models and Data

jcApp.rawdata = {"Ni1.17, Co0.28, Si0.45, Cr0.09, SHT=Pilot, aging=460 degC": {"input": [1.17, 0.28, 0.45, 0.09, 0.0, 1.0, 0.0, 0.0, 0.0, 460.0], "time [min]": [180], "conductivity [%IACS]": [43.9], "hardness [HV]": [192]}, "Ni1.17, Co0.28, Si0.45, Cr0.09, SHT=Pilot, aging=480 degC": {"input": [1.17, 0.28, 0.45, 0.09, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [300], "conductivity [%IACS]": [45.8], "hardness [HV]": [185]}, "Ni1.17, Co0.28, Si0.45, Cr0.09, SHT=Pilot, aging=500 degC": {"input": [1.17, 0.28, 0.45, 0.09, 0.0, 1.0, 0.0, 0.0, 0.0, 500.0], "time [min]": [120], "conductivity [%IACS]": [45.7], "hardness [HV]": [186]}, "Ni1.17, Co0.28, Si0.45, Cr0.09, SHT=Pilot, 1st aging=460 degC, 180 min, aging=380 degC": {"input": [1.17, 0.28, 0.45, 0.09, 0.0, 1.0, 0.0, 460.0, 180.0, 380.0], "time [min]": [180], "conductivity [%IACS]": [46.1], "hardness [HV]": [233]}, "Ni1.17, Co0.28, Si0.45, Cr0.09, SHT=Pilot, 1st aging=460 degC, 180 min, aging=400 degC": {"input": [1.17, 0.28, 0.45, 0.09, 0.0, 1.0, 0.0, 460.0, 180.0, 400.0], "time [min]": [180, 600], "conductivity [%IACS]": [45.2, 46.0], "hardness [HV]": [228, 220]}, "Ni1.17, Co0.28, Si0.45, Cr0.09, SHT=Pilot, 1st aging=480 degC, 300 min, aging=380 degC": {"input": [1.17, 0.28, 0.45, 0.09, 0.0, 1.0, 0.0, 480.0, 300.0, 380.0], "time [min]": [180], "conductivity [%IACS]": [48.6], "hardness [HV]": [216]}, "Ni1.17, Co0.28, Si0.45, Cr0.09, SHT=Pilot, 1st aging=480 degC, 300 min, aging=400 degC": {"input": [1.17, 0.28, 0.45, 0.09, 0.0, 1.0, 0.0, 480.0, 300.0, 400.0], "time [min]": [600, 180], "conductivity [%IACS]": [48.1, 48.1], "hardness [HV]": [208, 211]}, "Ni1.17, Co0.28, Si0.45, Cr0.09, SHT=Pilot, 1st aging=500 degC, 120 min, aging=380 degC": {"input": [1.17, 0.28, 0.45, 0.09, 0.0, 1.0, 0.0, 500.0, 120.0, 380.0], "time [min]": [180], "conductivity [%IACS]": [48.0], "hardness [HV]": [214]}, "Ni1.17, Co0.28, Si0.45, Cr0.09, SHT=Pilot, 1st aging=500 degC, 120 min, aging=400 degC": {"input": [1.17, 0.28, 0.45, 0.09, 0.0, 1.0, 0.0, 500.0, 120.0, 400.0], "time [min]": [180, 600], "conductivity [%IACS]": [48.0, 48.4], "hardness [HV]": [215, 213]}, "Ni1.29, Co0.28, Si0.46, Cr0.09, SHT=Pilot, aging=480 degC": {"input": [1.29, 0.28, 0.46, 0.09, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [51.6], "hardness [HV]": [200]}, "Ni1.29, Co0.28, Si0.46, Cr0.09, SHT=Pilot, aging=500 degC": {"input": [1.29, 0.28, 0.46, 0.09, 0.0, 1.0, 0.0, 0.0, 0.0, 500.0], "time [min]": [360], "conductivity [%IACS]": [52.6], "hardness [HV]": [187]}, "Ni1.29, Co0.28, Si0.46, Cr0.09, SHT=Pilot, 1st aging=480 degC, 360 min, aging=380 degC": {"input": [1.29, 0.28, 0.46, 0.09, 0.0, 1.0, 0.0, 480.0, 360.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [57.7], "hardness [HV]": [218]}, "Ni1.29, Co0.28, Si0.46, Cr0.09, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.29, 0.28, 0.46, 0.09, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [60.3], "hardness [HV]": [205]}, "Ni1.29, Co0.28, Si0.46, Cr0.09, SHT=Pilot, 1st aging=500 degC, 360 min, aging=380 degC": {"input": [1.29, 0.28, 0.46, 0.09, 0.0, 1.0, 0.0, 500.0, 360.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [60.5], "hardness [HV]": [203]}, "Ni1.29, Co0.28, Si0.46, Cr0.09, SHT=Pilot, 1st aging=500 degC, 360 min, aging=400 degC": {"input": [1.29, 0.28, 0.46, 0.09, 0.0, 1.0, 0.0, 500.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [60.7], "hardness [HV]": [203]}, "Ni1.32, Co0.25, Si0.37, Cr0.04, SHT=Pilot, aging=480 degC": {"input": [1.32, 0.25, 0.37, 0.04, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [51.1], "hardness [HV]": [191]}, "Ni1.32, Co0.25, Si0.37, Cr0.04, SHT=Pilot, aging=500 degC": {"input": [1.32, 0.25, 0.37, 0.04, 0.0, 1.0, 0.0, 0.0, 0.0, 500.0], "time [min]": [360], "conductivity [%IACS]": [52.1], "hardness [HV]": [180]}, "Ni1.32, Co0.25, Si0.37, Cr0.04, SHT=Pilot, 1st aging=480 degC, 360 min, aging=380 degC": {"input": [1.32, 0.25, 0.37, 0.04, 0.0, 1.0, 0.0, 480.0, 360.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [56.1], "hardness [HV]": [223]}, "Ni1.32, Co0.25, Si0.37, Cr0.04, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.32, 0.25, 0.37, 0.04, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [56.4], "hardness [HV]": [216]}, "Ni1.32, Co0.25, Si0.37, Cr0.04, SHT=Pilot, 1st aging=500 degC, 360 min, aging=380 degC": {"input": [1.32, 0.25, 0.37, 0.04, 0.0, 1.0, 0.0, 500.0, 360.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [59.0], "hardness [HV]": [200]}, "Ni1.32, Co0.25, Si0.37, Cr0.04, SHT=Pilot, 1st aging=500 degC, 360 min, aging=400 degC": {"input": [1.32, 0.25, 0.37, 0.04, 0.0, 1.0, 0.0, 500.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [60.2], "hardness [HV]": [202]}, "Ni1.35, Co0.22, Si0.38, Cr0.15, SHT=Pilot, aging=480 degC": {"input": [1.35, 0.22, 0.38, 0.15, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [52.6], "hardness [HV]": [185]}, "Ni1.35, Co0.22, Si0.38, Cr0.15, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.35, 0.22, 0.38, 0.15, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [59.6], "hardness [HV]": [212]}, "Ni1.35, Co0.28, Si0.45, Cr0.13, SHT=Pilot, aging=480 degC": {"input": [1.35, 0.28, 0.45, 0.13, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [48.3], "hardness [HV]": [190]}, "Ni1.35, Co0.28, Si0.45, Cr0.13, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.35, 0.28, 0.45, 0.13, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [53.3], "hardness [HV]": [211]}, "Ni1.45, Co0.25, Si0.46, Cr0.10, SHT=Pilot, aging=480 degC": {"input": [1.45, 0.25, 0.46, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [48.1], "hardness [HV]": [208]}, "Ni1.45, Co0.25, Si0.46, Cr0.10, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.45, 0.25, 0.46, 0.1, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [53.4], "hardness [HV]": [220]}, "Ni1.47, Co0.26, Si0.35, Cr0.08, SHT=Pilot, aging=480 degC": {"input": [1.47, 0.26, 0.35, 0.08, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [52.0], "hardness [HV]": [191]}, "Ni1.47, Co0.26, Si0.35, Cr0.08, SHT=Pilot, aging=500 degC": {"input": [1.47, 0.26, 0.35, 0.08, 0.0, 1.0, 0.0, 0.0, 0.0, 500.0], "time [min]": [360], "conductivity [%IACS]": [53.5], "hardness [HV]": [183]}, "Ni1.47, Co0.26, Si0.35, Cr0.08, SHT=Pilot, 1st aging=480 degC, 360 min, aging=380 degC": {"input": [1.47, 0.26, 0.35, 0.08, 0.0, 1.0, 0.0, 480.0, 360.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [57.1], "hardness [HV]": [212]}, "Ni1.47, Co0.26, Si0.35, Cr0.08, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.47, 0.26, 0.35, 0.08, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [58.2], "hardness [HV]": [205]}, "Ni1.47, Co0.26, Si0.35, Cr0.08, SHT=Pilot, 1st aging=500 degC, 360 min, aging=380 degC": {"input": [1.47, 0.26, 0.35, 0.08, 0.0, 1.0, 0.0, 500.0, 360.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [60.1], "hardness [HV]": [201]}, "Ni1.47, Co0.26, Si0.35, Cr0.08, SHT=Pilot, 1st aging=500 degC, 360 min, aging=400 degC": {"input": [1.47, 0.26, 0.35, 0.08, 0.0, 1.0, 0.0, 500.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [59.5], "hardness [HV]": [200]}, "Ni1.52, Co0.30, Si0.43, Cr0.08, SHT=Pilot, aging=480 degC": {"input": [1.52, 0.3, 0.43, 0.08, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360, 600], "conductivity [%IACS]": [51.8, 52.5], "hardness [HV]": [199, 200]}, "Ni1.52, Co0.30, Si0.43, Cr0.08, SHT=Pilot, aging=500 degC": {"input": [1.52, 0.3, 0.43, 0.08, 0.0, 1.0, 0.0, 0.0, 0.0, 500.0], "time [min]": [360], "conductivity [%IACS]": [53.0], "hardness [HV]": [189]}, "Ni1.52, Co0.30, Si0.43, Cr0.08, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.52, 0.3, 0.43, 0.08, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [57.0], "hardness [HV]": [226]}, "Ni1.52, Co0.30, Si0.43, Cr0.08, SHT=Pilot, 1st aging=480 degC, 600 min, aging=400 degC": {"input": [1.52, 0.3, 0.43, 0.08, 0.0, 1.0, 0.0, 480.0, 600.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [58.4], "hardness [HV]": [220]}, "Ni1.52, Co0.30, Si0.43, Cr0.08, SHT=Pilot, 1st aging=480 degC, 600 min, aging=420 degC": {"input": [1.52, 0.3, 0.43, 0.08, 0.0, 1.0, 0.0, 480.0, 600.0, 420.0], "time [min]": [360], "conductivity [%IACS]": [56.0], "hardness [HV]": [218]}, "Ni1.52, Co0.30, Si0.43, Cr0.08, SHT=Pilot, 1st aging=500 degC, 360 min, aging=400 degC": {"input": [1.52, 0.3, 0.43, 0.08, 0.0, 1.0, 0.0, 500.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [60.3], "hardness [HV]": [213]}, "Ni1.52, Co0.30, Si0.43, Cr0.08, SHT=Pilot, 1st aging=500 degC, 360 min, aging=420 degC": {"input": [1.52, 0.3, 0.43, 0.08, 0.0, 1.0, 0.0, 500.0, 360.0, 420.0], "time [min]": [360], "conductivity [%IACS]": [57.2], "hardness [HV]": [210]}, "Ni1.56, Co0.29, Si0.45, Cr0.13, SHT=Pilot, aging=480 degC": {"input": [1.56, 0.29, 0.45, 0.13, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [50.9], "hardness [HV]": [196]}, "Ni1.56, Co0.29, Si0.45, Cr0.13, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.56, 0.29, 0.45, 0.13, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [57.0], "hardness [HV]": [221]}, "Ni1.56, Co0.31, Si0.41, Cr0.13, SHT=Pilot, aging=480 degC": {"input": [1.56, 0.31, 0.41, 0.13, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [52.8], "hardness [HV]": [200]}, "Ni1.56, Co0.31, Si0.41, Cr0.13, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.56, 0.31, 0.41, 0.13, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [60.4], "hardness [HV]": [212]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=Pilot, aging=480 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [47.3], "hardness [HV]": [210]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [53.1], "hardness [HV]": [223]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, aging=440 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 0.0, 0.0, 440.0], "time [min]": [120, 480, 240, 360], "conductivity [%IACS]": [38.675, 42.4, 40.85, 41.85], "hardness [HV]": [206, 211, 208, 209]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, aging=460 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 0.0, 0.0, 460.0], "time [min]": [120, 240, 480, 180, 360], "conductivity [%IACS]": [39.875, 41.525, 43.0, 40.875, 42.475], "hardness [HV]": [207, 206, 204, 207, 205]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, aging=480 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 0.0, 0.0, 480.0], "time [min]": [480, 360, 240, 120, 300], "conductivity [%IACS]": [43.55, 42.85, 42.025, 40.625, 42.7], "hardness [HV]": [198, 197, 201, 204, 200]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, aging=500 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 0.0, 0.0, 500.0], "time [min]": [480, 240, 120, 360], "conductivity [%IACS]": [43.3, 41.975, 40.425, 42.725], "hardness [HV]": [180, 190, 200, 184]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=440 degC, 120 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 440.0, 120.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [47.5], "hardness [HV]": [223]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=440 degC, 120 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 440.0, 120.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [51.05], "hardness [HV]": [197]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=440 degC, 240 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 440.0, 240.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [47.8], "hardness [HV]": [230]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=440 degC, 240 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 440.0, 240.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [51.0], "hardness [HV]": [195]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=440 degC, 360 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 440.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [49.475], "hardness [HV]": [224]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=440 degC, 360 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 440.0, 360.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [51.65], "hardness [HV]": [193]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=440 degC, 480 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 440.0, 480.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [50.55], "hardness [HV]": [222]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=440 degC, 480 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 440.0, 480.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [48.9], "hardness [HV]": [232]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 120 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 120.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [51.85], "hardness [HV]": [215]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 120 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 120.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [48.95], "hardness [HV]": [213]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 180 min, aging=380 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 180.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [48.8], "hardness [HV]": [251]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 180 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 180.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [48.775], "hardness [HV]": [234]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 240 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 240.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [49.925], "hardness [HV]": [230]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 240 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 240.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [51.425], "hardness [HV]": [211]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 360 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [50.8], "hardness [HV]": [231]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 360 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 360.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [51.45], "hardness [HV]": [217]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 480 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 480.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [51.325], "hardness [HV]": [224]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=460 degC, 480 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 460.0, 480.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [51.85], "hardness [HV]": [215]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 120 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 120.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [50.725], "hardness [HV]": [224]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 120 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 120.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [51.4], "hardness [HV]": [196]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 240 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 240.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [52.225], "hardness [HV]": [227]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 240 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 240.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [51.825], "hardness [HV]": [208]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 300 min, aging=380 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 300.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [52.425], "hardness [HV]": [230]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 300 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 300.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [52.475], "hardness [HV]": [225]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [53.275], "hardness [HV]": [222]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 360 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 360.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [52.925], "hardness [HV]": [203]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 480 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 480.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [53.775], "hardness [HV]": [219]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=480 degC, 480 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 480.0, 480.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [53.475], "hardness [HV]": [204]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=500 degC, 120 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 500.0, 120.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [52.65], "hardness [HV]": [225]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=500 degC, 120 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 500.0, 120.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [51.975], "hardness [HV]": [222]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=500 degC, 240 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 500.0, 240.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [54.075], "hardness [HV]": [209]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=500 degC, 240 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 500.0, 240.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [53.525], "hardness [HV]": [210]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=500 degC, 360 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 500.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [54.675], "hardness [HV]": [204]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=500 degC, 360 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 500.0, 360.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [54.725], "hardness [HV]": [201]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=500 degC, 480 min, aging=400 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 500.0, 480.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [55.25], "hardness [HV]": [198]}, "Ni1.71, Co0.32, Si0.47, Cr0.18, SHT=AP7, 1st aging=500 degC, 480 min, aging=420 degC": {"input": [1.71, 0.32, 0.47, 0.18, 0.0, 0.0, 1.0, 500.0, 480.0, 420.0], "time [min]": [600], "conductivity [%IACS]": [55.65], "hardness [HV]": [195]}, "Ni1.71, Co0.33, Si0.41, Cr0.14, SHT=Pilot, aging=480 degC": {"input": [1.71, 0.33, 0.41, 0.14, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360, 600], "conductivity [%IACS]": [50.8, 51.7], "hardness [HV]": [204, 204]}, "Ni1.71, Co0.33, Si0.41, Cr0.14, SHT=Pilot, aging=500 degC": {"input": [1.71, 0.33, 0.41, 0.14, 0.0, 1.0, 0.0, 0.0, 0.0, 500.0], "time [min]": [360], "conductivity [%IACS]": [51.9], "hardness [HV]": [196]}, "Ni1.71, Co0.33, Si0.41, Cr0.14, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.71, 0.33, 0.41, 0.14, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [54.4], "hardness [HV]": [226]}, "Ni1.71, Co0.33, Si0.41, Cr0.14, SHT=Pilot, 1st aging=480 degC, 600 min, aging=400 degC": {"input": [1.71, 0.33, 0.41, 0.14, 0.0, 1.0, 0.0, 480.0, 600.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [57.6], "hardness [HV]": [220]}, "Ni1.71, Co0.33, Si0.41, Cr0.14, SHT=Pilot, 1st aging=480 degC, 600 min, aging=420 degC": {"input": [1.71, 0.33, 0.41, 0.14, 0.0, 1.0, 0.0, 480.0, 600.0, 420.0], "time [min]": [360], "conductivity [%IACS]": [56.0], "hardness [HV]": [217]}, "Ni1.71, Co0.33, Si0.41, Cr0.14, SHT=Pilot, 1st aging=500 degC, 360 min, aging=400 degC": {"input": [1.71, 0.33, 0.41, 0.14, 0.0, 1.0, 0.0, 500.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [58.9], "hardness [HV]": [215]}, "Ni1.71, Co0.33, Si0.41, Cr0.14, SHT=Pilot, 1st aging=500 degC, 360 min, aging=420 degC": {"input": [1.71, 0.33, 0.41, 0.14, 0.0, 1.0, 0.0, 500.0, 360.0, 420.0], "time [min]": [360], "conductivity [%IACS]": [56.9], "hardness [HV]": [217]}, "Ni1.72, Co0.34, Si0.48, Cr0.20, SHT=Pilot, aging=480 degC": {"input": [1.72, 0.34, 0.48, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [47.4], "hardness [HV]": [205]}, "Ni1.72, Co0.34, Si0.48, Cr0.20, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.72, 0.34, 0.48, 0.2, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [51.9], "hardness [HV]": [229]}, "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, aging=460 degC": {"input": [1.74, 0.28, 0.41, 0.13, 0.06, 1.0, 0.0, 0.0, 0.0, 460.0], "time [min]": [180], "conductivity [%IACS]": [47.1], "hardness [HV]": [205]}, "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, aging=480 degC": {"input": [1.74, 0.28, 0.41, 0.13, 0.06, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [300], "conductivity [%IACS]": [47.1], "hardness [HV]": [205]}, "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, aging=500 degC": {"input": [1.74, 0.28, 0.41, 0.13, 0.06, 1.0, 0.0, 0.0, 0.0, 500.0], "time [min]": [120], "conductivity [%IACS]": [51.5], "hardness [HV]": [196]}, "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, 1st aging=460 degC, 180 min, aging=380 degC": {"input": [1.74, 0.28, 0.41, 0.13, 0.06, 1.0, 0.0, 460.0, 180.0, 380.0], "time [min]": [180], "conductivity [%IACS]": [50.5], "hardness [HV]": [242]}, "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, 1st aging=460 degC, 180 min, aging=400 degC": {"input": [1.74, 0.28, 0.41, 0.13, 0.06, 1.0, 0.0, 460.0, 180.0, 400.0], "time [min]": [600, 180], "conductivity [%IACS]": [53.5, 51.1], "hardness [HV]": [200, 232]}, "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, 1st aging=480 degC, 300 min, aging=380 degC": {"input": [1.74, 0.28, 0.41, 0.13, 0.06, 1.0, 0.0, 480.0, 300.0, 380.0], "time [min]": [180], "conductivity [%IACS]": [54.3], "hardness [HV]": [223]}, "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, 1st aging=480 degC, 300 min, aging=400 degC": {"input": [1.74, 0.28, 0.41, 0.13, 0.06, 1.0, 0.0, 480.0, 300.0, 400.0], "time [min]": [600, 180], "conductivity [%IACS]": [56.2, 54.7], "hardness [HV]": [205, 217]}, "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, 1st aging=500 degC, 120 min, aging=380 degC": {"input": [1.74, 0.28, 0.41, 0.13, 0.06, 1.0, 0.0, 500.0, 120.0, 380.0], "time [min]": [180], "conductivity [%IACS]": [54.0], "hardness [HV]": [220]}, "Ni1.74, Co0.28, Si0.41, Cr0.13, Mn0.06, SHT=Pilot, 1st aging=500 degC, 120 min, aging=400 degC": {"input": [1.74, 0.28, 0.41, 0.13, 0.06, 1.0, 0.0, 500.0, 120.0, 400.0], "time [min]": [180, 600], "conductivity [%IACS]": [54.4, 55.4], "hardness [HV]": [224, 210]}, "Ni1.77, Co0.31, Si0.45, Cr0.13, SHT=Pilot, aging=480 degC": {"input": [1.77, 0.31, 0.45, 0.13, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [51.8], "hardness [HV]": [208]}, "Ni1.77, Co0.31, Si0.45, Cr0.13, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.77, 0.31, 0.45, 0.13, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [59.1], "hardness [HV]": [227]}, "Ni1.77, Co0.31, Si0.45, Cr0.13, Mn0.13, SHT=Pilot, aging=480 degC": {"input": [1.77, 0.31, 0.45, 0.13, 0.13, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [45.2], "hardness [HV]": [228]}, "Ni1.77, Co0.31, Si0.45, Cr0.13, Mn0.13, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.77, 0.31, 0.45, 0.13, 0.13, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [50.3], "hardness [HV]": [210]}, "Ni1.78, Co0.29, Si0.45, Cr0.12, SHT=Pilot, aging=480 degC": {"input": [1.78, 0.29, 0.45, 0.12, 0.0, 1.0, 0.0, 0.0, 0.0, 480.0], "time [min]": [360], "conductivity [%IACS]": [49.6], "hardness [HV]": [211]}, "Ni1.78, Co0.29, Si0.45, Cr0.12, SHT=Pilot, aging=500 degC": {"input": [1.78, 0.29, 0.45, 0.12, 0.0, 1.0, 0.0, 0.0, 0.0, 500.0], "time [min]": [360], "conductivity [%IACS]": [51.9], "hardness [HV]": [198]}, "Ni1.78, Co0.29, Si0.45, Cr0.12, SHT=Pilot, 1st aging=480 degC, 360 min, aging=380 degC": {"input": [1.78, 0.29, 0.45, 0.12, 0.0, 1.0, 0.0, 480.0, 360.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [57.8], "hardness [HV]": [222]}, "Ni1.78, Co0.29, Si0.45, Cr0.12, SHT=Pilot, 1st aging=480 degC, 360 min, aging=400 degC": {"input": [1.78, 0.29, 0.45, 0.12, 0.0, 1.0, 0.0, 480.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [58.5], "hardness [HV]": [206]}, "Ni1.78, Co0.29, Si0.45, Cr0.12, SHT=Pilot, 1st aging=500 degC, 360 min, aging=380 degC": {"input": [1.78, 0.29, 0.45, 0.12, 0.0, 1.0, 0.0, 500.0, 360.0, 380.0], "time [min]": [600], "conductivity [%IACS]": [58.5], "hardness [HV]": [203]}, "Ni1.78, Co0.29, Si0.45, Cr0.12, SHT=Pilot, 1st aging=500 degC, 360 min, aging=400 degC": {"input": [1.78, 0.29, 0.45, 0.12, 0.0, 1.0, 0.0, 500.0, 360.0, 400.0], "time [min]": [600], "conductivity [%IACS]": [60.7], "hardness [HV]": [201]}};

