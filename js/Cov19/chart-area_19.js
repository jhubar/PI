// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';



loadData();


function loadData(){

    var data_cov19_be = ''
    // DAp
    var tmp ;



      const result = $data_cov19_be;


      data_cov19_be_date = [];
      data_cov19_be_cases = [];
      data_cov19_be_age_group = [];
      data_cov19_be_region = [];
      data_cov19_be_sex = [];
      data_cov19_be_province = [];



      for(var i=0;i<result.length-10;i++){

        data_cov19_be_date.push(result[i].DATE);
        data_cov19_be_province.push(result[i].PROVINCE);
        data_cov19_be_region.push(result[i].REGION);
        data_cov19_be_age_group.push(result[i].AGEGROUP);
        data_cov19_be_sex.push(result[i].SEX);
        data_cov19_be_cases.push(result[i].CASES);

      }


      data_processing();


  var ctx_cov_19 = document.getElementById("myAreaCOV19");
  draw_current_data19();

}


function data_processing(){




  const mapper = single => {
  let d = single.DATE.split('-');
  let p = Number(single.CASES);
  return { year: d[0], month: d[1], CASES: p };
}

const reducer = (group, current) => {
  let i = group.findIndex(single => (single.year == current.year && single.month == current.month));
  if (i == -1) {
    return [ ...group, current ];
  }

  group[i].CASES += current.CASES;
  return group;
};

const sum_cases_per_month = $data_cov19_be.map(mapper).reduce(reducer, []);
console.log(Object.values(sum_cases_per_month));

month = [];
sum_cases = [];


for(var i=0;i<sum_cases_per_month.length-3;i++){

  if( sum_cases_per_month[i].CASES != 190 || sum_cases_per_month[i].year != "NA"){
  month.push(sum_cases_per_month[i].month);
  sum_cases.push(sum_cases_per_month[i].CASES);
  }


}

console.log(month)


}
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


function draw_current_data19() {

  if (typeof(myLineChart_cov_19) != "undefined"){

    myLineChart_positive.destroy();

  }

  ctx_cov_19 = document.getElementById("myAreaCOV19");
  myLineChart_cov_19 = new Chart(ctx_cov_19, {
    type: 'line',
    data: {
      labels: month,
      datasets: [
        // positives
        {
          label: "Positives ",
          lineTension: 0.6,
          backgroundColor: "rgba(78, 115, 223, 0.2)",
          borderColor: "rgba(78, 115, 223, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(78, 115, 223, 1)",
          pointBorderColor: "rgba(78, 115, 223, 1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
          pointHoverBorderColor: "rgba(78, 115, 223, 1)",
          pointHitRadius: 1,
          pointBorderWidth: 1,
          data: sum_cases,
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
