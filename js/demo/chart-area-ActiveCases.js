// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

const $value_spreading_period = $('.value_spreading_period');
const $value = $('#range_spreading_period');
$value_spreading_period.html($value.val());

const $value_time_period = $('.value_time_period');
const $value_time = $('#range_time_period');
$value_time_period.html($value_time.val());
loadData();
$value_time.on('input change', () => {
  // loadData();
  $value_spreading_period.html($value.val());
  $value_time_period.html($value_time.val());

  var url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
  var data = ''
  // DAp
  var tmp ;
  $.get(url,function(data){
    var result = [];
    var lines=data.split("\n");
    var headers=lines[0].split(",");
    for(var i=1;i<lines.length;i++){
      var obj = {};
      var currentline=lines[i].split(",");
      for(var j=0;j<headers.length;j++){
        obj[headers[j]] = currentline[j];
      }
      result.push(obj);
    }


    label = [];
    dataL = [];
    dataC = [];
    dataOverfit = [];
    dataUnderfit = [];
    dataLinearfit = [];
    recovered_population =[];
    for(var i=0;i<result.length-1;i++){
      label.push(result[i].Day);
      dataL.push(result[i].num_positive);
      dataOverfit.push(result[i].num_positive);
      dataUnderfit.push(result[i].num_positive);
      dataLinearfit.push(result[i].num_positive);
      if(i >= $value.val() && dataL[i-1] >=0){
        dataC.push((parseInt(result[i].num_positive) + parseInt(dataC[i-1]) - parseInt(dataLinearfit[i-$value.val()+1]) ).toString());
        recovered_population.push(parseInt(dataLinearfit[i-$value.val()+1]).toString());
      }
      else if (i != 0) {
        dataC.push((parseInt(result[i].num_positive) + parseInt(dataC[i-1])).toString());
        recovered_population.push("0");
      }
      else {
        dataC.push(result[i].num_positive);
        recovered_population.push("0");
      }
    }



    for(var i=0;i<$value_time.val();i++){
      label.push((result.length+i).toString());
    }



    tmpOverfit = dataL[result.length-2];
    tmpUnderfit = dataL[result.length-2];
    tmpLinearfit = dataL[result.length-2];

    var m = 0;
    for(var i=1;i<result.length-1;i++){
        m +=(dataL[i] - dataL[i-1])/((label[i] - label[i-1]));
    }
    m/=result.length-1;

    for(var i=result.length-1;i<label.length;i++){
      tmpLinearfit = m * parseInt(label[i]);
      if(i%2 == 0){
        tmpOverfit = 4 + parseInt(tmpLinearfit);
        tmpUnderfit = parseInt(tmpLinearfit) - 4;
      }
      else{
        tmpOverfit = 3 + parseInt(tmpLinearfit);
        tmpUnderfit = parseInt(tmpLinearfit)-3;
      }


      dataLinearfit[i]= tmpLinearfit.toString();
      dataOverfit[i]= tmpOverfit.toString()*1.1;
      dataUnderfit[i]= tmpUnderfit.toString()/1.1;

      pred_cum = (parseInt(tmpLinearfit) + parseInt(dataC[i-1]) - parseInt(dataLinearfit[i-$value.val()]))

      if ( pred_cum >=  0 ){
        dataC.push(pred_cum.toString());
        recovered_population.push(dataLinearfit[i-$value.val()]);

      }
    }
//


  var ctx_active_cases = document.getElementById("myAreaChart");



  without_cum_cases(2);

  $("#num_Of_cum_casesKPIForcast").html(dataC[dataC.length-1])
  var percentChange = ((((dataLinearfit[label.length-1] - dataLinearfit[label.length-8]) /dataLinearfit[label.length-8]))*100).toFixed(0);

  if(percentChange == 0){
    result = `


    <td class="fas fa-caret-right text-success">${ percentChange }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_OfcasesKPIPredicated").html(result)
  }
  if(percentChange <= 0){
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_OfcasesKPIPredicated").html(result)
  }
  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_OfcasesKPIPredicated").html(result)
  }
  var currentforcastResult = (parseInt(dataLinearfit[label.length-1]).toFixed(0)).toString();

  forcastResult = `
  <td class="text-secondary">${ currentforcastResult }</td>`
  $("#num_OfcasesKPIForcast").html(forcastResult)

  var num_recovered_cases = (parseInt(dataLinearfit[label.length-1]).toFixed(0)).toString();
  num_recovered_cases = `
  <td class="text-secondary">${ currentforcastResult }</td>`
  $("#num_OfcasesKPIForcast").html(num_recovered_cases)



},
);




});



$value.on('input change', () => {
  // loadData();
  $value_spreading_period.html($value.val());
  $value_time_period.html($value_time.val());

  var url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
  var data = ''
  // DAp
  var tmp ;
  $.get(url,function(data){
    var result = [];
    var lines=data.split("\n");
    var headers=lines[0].split(",");
    for(var i=1;i<lines.length;i++){
      var obj = {};
      var currentline=lines[i].split(",");
      for(var j=0;j<headers.length;j++){
        obj[headers[j]] = currentline[j];
      }
      result.push(obj);
    }


    label = [];
    dataL = [];
    dataC = [];
    dataOverfit = [];
    dataUnderfit = [];
    dataLinearfit = [];
    recovered_population =[];
    for(var i=0;i<result.length-1;i++){
      label.push(result[i].Day);
      dataL.push(result[i].num_positive);
      dataOverfit.push(result[i].num_positive);
      dataUnderfit.push(result[i].num_positive);
      dataLinearfit.push(result[i].num_positive);
      if(i >= $value.val() && dataL[i-1] >=0){
        dataC.push((parseInt(result[i].num_positive) + parseInt(dataC[i-1]) - parseInt(dataLinearfit[i-$value.val()+1]) ).toString());
        recovered_population.push(parseInt(dataLinearfit[i-$value.val()+1]).toString());
      }
      else if (i != 0) {
        dataC.push((parseInt(result[i].num_positive) + parseInt(dataC[i-1])).toString());
        recovered_population.push("0");
      }
      else {
        dataC.push(result[i].num_positive);
        recovered_population.push("0");
      }
    }



    for(var i=0;i<$value_time.val();i++){
      label.push((result.length+i).toString());
    }



    tmpOverfit = dataL[result.length-2];
    tmpUnderfit = dataL[result.length-2];
    tmpLinearfit = dataL[result.length-2];

    var m = 0;
    for(var i=1;i<result.length-1;i++){
        m +=(dataL[i] - dataL[i-1])/((label[i] - label[i-1]));
    }
    m/=result.length-1;

    for(var i=result.length-1;i<label.length;i++){
      tmpLinearfit = m * parseInt(label[i]);
      if(i%2 == 0){
        tmpOverfit = 4 + parseInt(tmpLinearfit);
        tmpUnderfit = parseInt(tmpLinearfit) - 4;
      }
      else{
        tmpOverfit = 3 + parseInt(tmpLinearfit);
        tmpUnderfit = parseInt(tmpLinearfit)-3;
      }


      dataLinearfit[i]= tmpLinearfit.toString();
      dataOverfit[i]= tmpOverfit.toString()*1.1;
      dataUnderfit[i]= tmpUnderfit.toString()/1.1;

      pred_cum = (parseInt(tmpLinearfit) + parseInt(dataC[i-1]) - parseInt(dataLinearfit[i-$value.val()]))

      if ( pred_cum >=  0 ){
        dataC.push(pred_cum.toString());
        recovered_population.push(dataLinearfit[i-$value.val()]);

      }
    }
//


  var ctx_active_cases = document.getElementById("myAreaChart");



  without_cum_cases(2);

  $("#num_Of_cum_casesKPIForcast").html(dataC[dataC.length-1])
  var percentChange = ((((dataLinearfit[label.length-1] - dataLinearfit[label.length-8]) /dataLinearfit[label.length-8]))*100).toFixed(0);

  if(percentChange == 0){
    result = `


    <td class="fas fa-caret-right text-success">${ percentChange }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_OfcasesKPIPredicated").html(result)
  }
  if(percentChange <= 0){
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_OfcasesKPIPredicated").html(result)
  }
  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_OfcasesKPIPredicated").html(result)
  }
  var currentforcastResult = (parseInt(dataLinearfit[label.length-1]).toFixed(0)).toString();

  forcastResult = `
  <td class="text-secondary">${ currentforcastResult }</td>`
  $("#num_OfcasesKPIForcast").html(forcastResult)


},
);




});

function number_format(number, decimals, dec_point, thousands_sep) {
  // *     example: number_format(1234.56, 2, ',', ' ');
  // *     return: '1 234,56'
  number = (number + '').replace(',', '').replace(' ', '');
  var n = !isFinite(+number) ? 0 : +number,
    prec = !isFinite(+decimals) ? 0 : Math.abs(decimals),
    sep = (typeof thousands_sep === 'undefined') ? ',' : thousands_sep,
    dec = (typeof dec_point === 'undefined') ? '.' : dec_point,
    s = '',
    toFixedFix = function(n, prec) {
      var k = Math.pow(10, prec);
      return '' + Math.round(n * k) / k;
    };
  // Fix for IE parseFloat(0.55).toFixed(0) = 0;
  s = (prec ? toFixedFix(n, prec) : '' + Math.round(n)).split('.');
  if (s[0].length > 3) {
    s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep);
  }
  if ((s[1] || '').length < prec) {
    s[1] = s[1] || '';
    s[1] += new Array(prec - s[1].length + 1).join('0');
  }
  return s.join(dec);
}

function without_cum_cases(ans) {

  if (typeof(myLineChart) != "undefined"){

    myLineChart.destroy();

  }

  ctx_active_cases = document.getElementById("myAreaChart");
  myLineChart = new Chart(ctx_active_cases, {
    type: 'line',
    data: {
      labels: label,
      datasets: [
        // recovered_population
        {
          label: "Recovered cases ",
          lineTension: 0.6,
          backgroundColor: "rgba( 133, 135, 150 , 0.2)",
          borderColor: "rgba( 133, 135, 150 , 0.2)",
          pointRadius: 4,
          pointBackgroundColor: "rgba( 133, 135, 150 , 0.2)",
          pointBorderColor: "rgba( 133, 135, 150 , 0.2)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba( 133, 135, 150 , 0.2)",
          pointHoverBorderColor: "rgba( 133, 135, 150 , 0.2)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: recovered_population,
        },
        //cumulatives cases
        {
          label: "Cumulative ",
          lineTension: 0.6,
          backgroundColor: "rgba(34,139,34, 0.2)",
          borderColor: "rgba(34,139,34, 0.1)",
          pointRadius: 4,
          pointBackgroundColor: "rgba(34,139,34, 0.1)",
          pointBorderColor: "rgba(34,139,34, 0.1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(34,139,34, 0.1)",
          pointHoverBorderColor: "rgba(34,139,34, 0.1)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: cum_cases(ans,dataC),
        },
        // current cases
        {
          label: "Current ",
          lineTension: 0.6,
          backgroundColor: "rgba(78, 115, 223, 0.2)",
          borderColor: "rgba(78, 115, 223, 1)",
          pointRadius: 4,
          pointBackgroundColor: "rgba(78, 115, 223, 1)",
          pointBorderColor: "rgba(78, 115, 223, 1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
          pointHoverBorderColor: "rgba(78, 115, 223, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: dataL,
        },
        //Underfit line
        {
          label: "Underfit ",
          lineTension: 0.1,
          backgroundColor: "rgba(255,255,255,1)",
          borderColor: "rgba(255, 193, 7,0.1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(255, 193, 7,1)",
          pointBorderColor: "rgba(255, 193, 7,1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(255, 193, 7, 1)",
          pointHoverBorderColor: "rgba(255, 193, 7, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: dataUnderfit,
        },
        // linear line
        {
          label: "Linear ",
          lineTension: 0.3,
          backgroundColor: "rgba(255, 193, 7,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: dataLinearfit,
        },
        //Overfitt line
        {
          label: "Overfit ",
          lineTension: 0.1,
          backgroundColor: "rgba(255, 193, 7,0.3)",
          borderColor: "rgba(255, 193, 7,0.1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(255, 193, 7,1)",
          pointBorderColor: "rgba(255, 193, 7,1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(255, 193, 7, 1)",
          pointHoverBorderColor: "rgba(255, 193, 7, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: dataOverfit,
        }




    ],
    },
    options: {
      maintainAspectRatio: false,
      layout: {
        padding: {
          left: 10,
          right: 25,
          top: 25,
          bottom: 0
        }
      },
      scales: {
        xAxes: [{
          time: {
            unit: 'date'
          },
          gridLines: {
            display: false,
            drawBorder: false
          },
          ticks: {
            maxTicksLimit: 7,
            // callback: function(value, index, values) {
            //   return number_format(value)+ ' Days';
            // }
          }
        }],
        yAxes: [{
          ticks: {
            maxTicksLimit: 5,
            padding: 10,

            callback: function(value, index, values) {
              return number_format(value)+ ' Cases';
            }
          },
          gridLines: {
            color: "rgb(234, 236, 244)",
            zeroLineColor: "rgb(234, 236, 244)",
            drawBorder: false,
            borderDash: [2],
            zeroLineBorderDash: [2]
          }
        }],
      },
      legend: {
        display: false
      },
      tooltips: {
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        titleMarginBottom: 10,
        titleFontColor: '#6e707e',
        titleFontSize: 14,
        borderColor: '#dddfeb',
        borderWidth: 1,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        intersect: false,
        mode: 'index',
        caretPadding: 10,
        callbacks: {
          label: function(tooltipItem, chart) {
            var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
            return datasetLabel + number_format(tooltipItem.yLabel)+ ': Cases';
          }
        }
      }
    }
  },);


}


function cum_cases(ans, dataC){
  if(ans == 2){
    return dataC;
  }
  if(ans == 1){
    return [];
  }

}

function loadData(){
  var url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
  var data = ''
  // DAp
  var tmp ;
  $.get(url,function(data){
    var result = [];
    var lines=data.split("\n");
    var headers=lines[0].split(",");
    for(var i=1;i<lines.length;i++){
      var obj = {};
      var currentline=lines[i].split(",");
      for(var j=0;j<headers.length;j++){
        obj[headers[j]] = currentline[j];
      }
      result.push(obj);
    }


    label = [];
    dataL = [];
    dataC = [];
    dataOverfit = [];
    dataUnderfit = [];
    dataLinearfit = [];
    recovered_population = [];
    for(var i=0;i<result.length-1;i++){
      label.push(result[i].Day);
      dataL.push(result[i].num_positive);
      dataOverfit.push(result[i].num_positive);
      dataUnderfit.push(result[i].num_positive);
      dataLinearfit.push(result[i].num_positive);
      if(i >= $value.val() && dataL[i-1] >=0){

        dataC.push((parseInt(result[i].num_positive) + parseInt(dataC[i-1]) - parseInt(dataLinearfit[i-$value.val()+1]) ).toString());
        recovered_population.push(parseInt(dataLinearfit[i-$value.val()+1]).toString());
      }
      else if (i != 0) {
        dataC.push((parseInt(result[i].num_positive) + parseInt(dataC[i-1])).toString());
        recovered_population.push("0");
      }
      else {
        dataC.push(result[i].num_positive);
        recovered_population.push("0");
      }
    }



    for(var i=0;i<$value_time.val();i++){
      label.push((result.length+i).toString());
    }



    tmpOverfit = dataL[result.length-2];
    tmpUnderfit = dataL[result.length-2];
    tmpLinearfit = dataL[result.length-2];
    beta =  0.5841185;
    gamma = 0.4158816;
    r0 = beta * (1/gamma);
    var m = 0;
    for(var i=1;i<result.length-1;i++){
        m +=(dataL[i] - dataL[i-1])/((label[i] - label[i-1]));
    }
    m/=result.length-1;

    for(var i=result.length-1;i<label.length;i++){
      tmpLinearfit = m * parseInt(label[i]);
      if(i%2 == 0){
        tmpOverfit = 4 + parseInt(tmpLinearfit);
        tmpUnderfit = parseInt(tmpLinearfit) - 4;
      }
      else{
        tmpOverfit = 3 + parseInt(tmpLinearfit);
        tmpUnderfit = parseInt(tmpLinearfit)-3;
      }


      dataLinearfit[i]= tmpLinearfit.toString();
      dataOverfit[i]= tmpOverfit.toString()*1.1;
      dataUnderfit[i]= tmpUnderfit.toString()/1.1;

      pred_cum = (parseInt(tmpLinearfit) + parseInt(dataC[i-1]) - parseInt(dataLinearfit[i-$value.val()]))

      if ( pred_cum >=  0 ){
        dataC.push(pred_cum.toString());
        recovered_population.push(dataLinearfit[i-$value.val()]);
      }
    }



  var ctx_active_cases = document.getElementById("myAreaChart");



  without_cum_cases(2);
  $("#num_Of_cum_casesKPIForcast").html(dataC[dataC.length-1])
  var percentChange = ((((dataLinearfit[label.length-1] - dataLinearfit[label.length-8]) /dataLinearfit[label.length-8]))*100).toFixed(0);

  if(percentChange == 0){
    result = `


    <td class="fas fa-caret-right text-success">${ percentChange }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_OfcasesKPIPredicated").html(result)
  }
  if(percentChange <= 0){
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_OfcasesKPIPredicated").html(result)
  }
  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_OfcasesKPIPredicated").html(result)
  }
  var currentforcastResult = (parseInt(dataLinearfit[label.length-1]).toFixed(0)).toString();

  forcastResult = `
  <td class="text-secondary">${ currentforcastResult }</td>`
  $("#num_OfcasesKPIForcast").html(forcastResult)

  var current_recovered_cases = (parseInt(recovered_population[label.length - parseInt($value.val()) ]).toFixed(0)).toString();
  num_recovered_cases = `
  <td class="text-secondary">${ current_recovered_cases }</td>`
  $("#num_recovered_cases").html(num_recovered_cases)



},
);}
