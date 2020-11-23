// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';
const $url_data = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
const $url = "https://raw.githubusercontent.com/julien1941/PI/master/Python/Data/SEIR%2B.json?token=AL3RLGIADYBRO4ZMGHGTH6C7YSNOY"
const $value_time_period_data = $('.value_time_period_data');
const $value_time_data = $('#range_time_period_data');

const $value_time_period_SEIR = $('.value_time_period_SEIR');
const $value_time_SEIR = $('#range_time_period_SEIR');

$value_time_period_data.html($value_time_data.val());
$value_time_period_SEIR.html($value_time_SEIR.val());

const $id_switch_positive = document.getElementById('customSwitches_Positive');
const $id_switch_hospitalized = document.getElementById('customSwitches_hospitalized');
const $id_switch_cum_hospitalized = document.getElementById('customSwitches_cum_hospitalized');
const $id_switch_criticals = document.getElementById('customSwitches_criticals');
const $id_switch_fatalies = document.getElementById('customSwitches_fatalities');

const $value_customSwitchesSusceptible = $('#customSwitchesSusceptible');

const $id_switch_Susceptible = document.getElementById('customSwitchesSusceptible');
const $id_switch_Exosed = document.getElementById('customSwitchesExposed');
const $id_switch_Infected = document.getElementById('customSwitchesInfectious');
const $id_switch_Recovered = document.getElementById('customSwitchesRecovered');
const $id_switch_Hospitalized = document.getElementById('customSwitchesHospitalized');
const $id_switch_seir_criticals = document.getElementById('customSwitchesSeirCriticales');
const $id_switch_seir_fatalities = document.getElementById('customSwitchesSeirFatalities');

const $id_switch_num_bed_hos = document.getElementById('customSwitches_num_bed_hos');
const $id_switch_num_bed_icu = document.getElementById('customSwitches_num_bed_icu');

$id_switch_positive.addEventListener('change',function(){
    draw_current_data();
});

$id_switch_hospitalized.addEventListener('change',function(){
    draw_current_data();
});

$id_switch_cum_hospitalized.addEventListener('change',function(){
    draw_current_data();
});

$id_switch_criticals.addEventListener('change',function(){
    draw_current_data();
});

$id_switch_fatalies.addEventListener('change',function(){
    draw_current_data();
});


$id_switch_Susceptible.addEventListener('change',function(){
    draw();

});
$id_switch_num_bed_hos.addEventListener('change',function(){
    draw();
    load_card_value_seir()
});
$id_switch_num_bed_icu.addEventListener('change',function(){
    draw();
    load_card_value_seir()
});


$id_switch_Exosed.addEventListener('change',function(){
    draw();

});

$id_switch_Infected.addEventListener('change',function(){
    draw();

});

$id_switch_Recovered.addEventListener('change',function(){
    draw();

});

$id_switch_Hospitalized.addEventListener('change',function(){
    draw();


});
$id_switch_seir_criticals.addEventListener('change',function(){
    draw();

});
$id_switch_seir_fatalities.addEventListener('change',function(){
    draw();

});


load_cur_Data();
loadData();


function loadData(){


    var data_seir = ''
    // DAp
    var tmp ;

    $.get($url,function(data_seir){

      const result = JSON.parse(data_seir);


      data_day_seir = [];
      data_seir_s = [];
      data_seir_e = [];
      data_seir_i = [];
      data_seir_r = [];
      data_seir_h = [];
      data_seir_h_bis = [];

      data_seir_c = [];
      data_seir_c_bis = [];

      data_seir_f = [];
      data_seir_f_bis = [];
      data_seir_fc_bis = [];
      data_seir_f_tot = [];

      data_num_bed_hospit = [];
      data_num_bed_icu = [];

      tmp_data_seir_f_bis = [];
      tmp_data_seir_fc_bis = [];


      for(var i=0;i<$value_time_SEIR.val();i++){
        data_num_bed_hospit.push("4586")
        data_num_bed_icu.push("174")
        data_day_seir.push(result.predict[i].predict_day);
        data_seir_s.push(result.predict[i].predict_S);
        data_seir_e.push(result.predict[i].predict_E);
        data_seir_i.push(result.predict[i].predict_I);
        data_seir_r.push(result.predict[i].predict_R);
        data_seir_h.push(result.predict[i].predict_H);
        data_seir_c.push(result.predict[i].predict_C);
        data_seir_f.push(result.predict[i].predict_F);


        if ($value_time_SEIR.val() < result.length-1 ){

          data_num_hospitalised.push(result[i].num_hospitalised);
          data_num_critical.push(result[i].num_critical)
          data_num_fatalities.push(result[i].num_fatalities);
        }

        if(parseFloat(result.predict[i].predict_day) >= 103.0){

            if(parseFloat(result.predict[i].predict_H) >= 4586.0){
            data_seir_h_bis.push("4586");
            tmp_data_seir_f_bis.push(parseFloat(result.predict[i].predict_H)-4586);
          }
          else{
            data_seir_h_bis.push(result.predict[i].predict_H);
            tmp_data_seir_f_bis.push(0);
          }

        }else{
            data_seir_h_bis.push(result.predict[i].predict_H);
            tmp_data_seir_f_bis.push(0);
        }

        if(parseFloat(result.predict[i].predict_day) >= 78.0){

            if(parseFloat(result.predict[i].predict_C) >= 174){
            data_seir_c_bis.push("174");
            tmp_data_seir_fc_bis.push(parseFloat(result.predict[i].predict_C)-174.0);
          }
          else{
            data_seir_c_bis.push(result.predict[i].predict_C);
            tmp_data_seir_fc_bis.push(0);
          }

        }else{
            data_seir_c_bis.push(result.predict[i].predict_C);
            tmp_data_seir_fc_bis.push(0);
        }
      }


      var myarray = tmp_data_seir_f_bis;
      var tmp_data_seir_f_bis = [];
      myarray.reduce(function(a,b,i) { return tmp_data_seir_f_bis[i] = a+b; },0);

      var myarray2 = tmp_data_seir_fc_bis;
      var tmp_data_seir_fc_bis = [];
      myarray2.reduce(function(a,b,i) { return tmp_data_seir_fc_bis[i] = a+b; },0);


      for(var i=0;i<$value_time_SEIR.val();i++){
        data_seir_f_tot.push((tmp_data_seir_f_bis[i]+parseFloat(data_seir_f[i])+tmp_data_seir_fc_bis[i]+parseFloat(data_seir_f[i])).toString())
        data_seir_f_bis.push((tmp_data_seir_f_bis[i]+parseFloat(data_seir_f[i])).toString())
        data_seir_fc_bis.push((tmp_data_seir_fc_bis[i]+parseFloat(data_seir_f[i])).toString())
      }



  var ctx_active_cases = document.getElementById("myAreaSeirModel");


  $("#id_beta_seir").html((parseFloat(result.model[0].beta).toFixed(6)).toString())
  $("#id_sigma_seir").html((parseFloat(result.model[0].sigma).toFixed(6)).toString())
  $("#id_gamma_seir").html((parseFloat(result.model[0].gamma).toFixed(6)).toString())
  $("#id_hp_seir").html((parseFloat(result.model[0].hp).toFixed(6)).toString())
  $("#id_hcr_seir").html((parseFloat(result.model[0].hcr).toFixed(6)).toString())



  draw();



  load_card_value_seir();




},
);}

$value_time_SEIR.on('input change', () => {

  $value_time_period_SEIR.html($value_time_SEIR.val());
  loadData();
});





function load_cur_Data(){

    var data = ''
    // DAp
    var tmp ;

    $.get($url_data,function(data){
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


      data_day = [];
      data_num_positive = [];
      data_num_tested = [];
      data_num_hospitalised = [];
      data_num_cumulative_hospitalizations = [];
      data_num_critical = [];
      data_num_fatalities = [];



      for(var i=0;i<$value_time_data.val();i++){
        data_day.push(result[i].Day);
        data_num_positive.push(result[i].num_positive);
        data_num_tested.push(result[i].num_tested);
        data_num_hospitalised.push(result[i].num_hospitalised);
        data_num_cumulative_hospitalizations.push(result[i].num_cumulative_hospitalizations);
        data_num_critical.push(result[i].num_critical)
        data_num_fatalities.push(result[i].num_fatalities);

      }







  var ctx_positive = document.getElementById("myAreaChartPositive");



  draw_current_data();

  load_card_value();
  load_card_kpi_value();






},
);}


function load_card_value_seir(){
  $("#num_Of_Susceptible_seir").html((parseFloat(data_seir_s[data_seir_s.length-1]).toFixed(2)).toString())
  $("#num_Of_Exposed_seir").html((parseFloat(data_seir_e[data_seir_e.length-1]).toFixed(2)).toString())
  $("#num_Of_infected_seir").html((parseFloat(data_seir_i[data_seir_i.length-1]).toFixed(2)).toString())
  $("#num_Of_Recovered_seir").html((parseFloat(data_seir_r[data_seir_r.length-1]).toFixed(2)).toString())


  $("#num_Of_fatalities_seir").html((parseFloat(data_seir_f[data_seir_f.length-1]).toFixed(2)).toString())
  $("#num_Of_day_seir").html(((parseFloat(data_day_seir[data_day_seir.length-1])+1).toFixed(0)).toString())

  if($id_switch_num_bed_hos.checked == true && $id_switch_num_bed_icu.checked == true){
    $("#num_Of_newFat").html((parseFloat(data_seir_f_tot[data_seir_f_tot.length-1]).toFixed(0)).toString());
    $("#num_Of_hospitalized_seir").html((parseFloat(data_seir_h_bis[data_seir_h_bis.length-1]).toFixed(2)).toString())
    $("#num_Of_criticals_seir").html((parseFloat(data_seir_c_bis[data_seir_c_bis.length-1]).toFixed(2)).toString())
  }
  else if($id_switch_num_bed_hos.checked == false && $id_switch_num_bed_icu.checked == true){
    $("#num_Of_newFat").html((parseFloat(data_seir_fc_bis[data_seir_fc_bis.length-1]).toFixed(0)).toString());
    $("#num_Of_hospitalized_seir").html((parseFloat(data_seir_h[data_seir_h.length-1]).toFixed(2)).toString())
    $("#num_Of_criticals_seir").html((parseFloat(data_seir_c_bis[data_seir_c_bis.length-1]).toFixed(2)).toString())
  }
  else if($id_switch_num_bed_hos.checked == true && $id_switch_num_bed_icu.checked == false){
    $("#num_Of_newFat").html((parseFloat(data_seir_f_bis[data_seir_f_bis.length-1]).toFixed(0)).toString());
    $("#num_Of_criticals_seir").html((parseFloat(data_seir_c[data_seir_c.length-1]).toFixed(2)).toString())
    $("#num_Of_hospitalized_seir").html((parseFloat(data_seir_h_bis[data_seir_h_bis.length-1]).toFixed(2)).toString())

  }else{
    $("#num_Of_newFat").html("Nan");
    $("#num_Of_hospitalized_seir").html((parseFloat(data_seir_h[data_seir_h.length-1]).toFixed(2)).toString())
    $("#num_Of_criticals_seir").html((parseFloat(data_seir_c[data_seir_c.length-1]).toFixed(2)).toString())
  }




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


function positives_draw(){
  if($id_switch_positive.checked == true){
    return data_num_positive;
  }
  else{
    return [];
  }
}
function hospitalized_draw(){
  if($id_switch_hospitalized.checked == true){
      return data_num_hospitalised;
  }
  else{
    return [];
  }
}
function cum_hospitalized_draw(){
  if($id_switch_cum_hospitalized.checked == true){
    return data_num_cumulative_hospitalizations;
  }
  else{
    return [];
  }
}

function criticals_draw(){
  if($id_switch_criticals.checked == true){
    return data_num_critical;
  }
  else{
    return [];
  }
}

function fatalities_draw(){
  if($id_switch_fatalies.checked == true){
    return data_num_fatalities;
  }
  else{
    return [];
  }
}
function susceptible_draw(){
  if($id_switch_Susceptible.checked == true){
    return data_seir_s;
  }
  else{
    return [];
  }
}
function exposed_draw(){
  if($id_switch_Exosed.checked == true){
    return data_seir_e;
  }
  else{
    return [];
  }
}

function infected_draw(){
  if($id_switch_Infected.checked == true){
    return data_seir_i;
  }
  else{
    return [];
  }
}

function recovered_draw(){
  if($id_switch_Recovered.checked == true){
    return data_seir_r;
  }
  else{
    return [];
  }
}
function hospitalized_seir_draw(){
  if($id_switch_Hospitalized.checked == true){
    if($id_switch_num_bed_hos.checked == false){
      return data_seir_h;
    }
    else{
      return data_seir_h_bis;
    }
  }
  else{
    return [];
  }
}
function criticals_seir_draw(){
  if($id_switch_seir_criticals.checked == true){
    if($id_switch_num_bed_icu.checked == false){
      return data_seir_c;
    }
    else{
      return data_seir_c_bis;
    }
  }
  else{
    return [];
  }
}
function fatalities_seir_draw(){
  if($id_switch_seir_fatalities.checked == true){
    return data_seir_f;
  }
  else{
    return [];
  }
}

function cum_bed_draw(){
  if($id_switch_num_bed_hos.checked == true){
    return data_num_bed_hospit;
  }
  else{
    return [];
  }
}
function cum_bed2_draw(){
  if($id_switch_seir_fatalities.checked == false){
    return [];
  }
  else if($id_switch_num_bed_hos.checked == true && $id_switch_num_bed_icu.checked == false){
    return data_seir_f_bis;
  }
  else if($id_switch_num_bed_icu.checked == true && $id_switch_num_bed_hos.checked == true){
    return data_seir_f_tot;
  }

  else{
    return [];
  }
}
function cum_icu_draw(){
  if($id_switch_num_bed_icu.checked == true){
    return data_num_bed_icu;
  }
  else{
    return [];
  }
}
function cum_icu2_draw(){
  if($id_switch_seir_fatalities.checked == false){
    return [];
  }
  else if($id_switch_num_bed_icu.checked == true && $id_switch_num_bed_hos.checked == false){
    return data_seir_fc_bis;
  }
  else if($id_switch_num_bed_icu.checked == true && $id_switch_num_bed_hos.checked == true){
    return data_seir_f_tot;
  }
  else{
    return [];
  }
}

$value_time_data.on('input change', () => {

  $value_time_period_data.html($value_time_data.val());
  load_cur_Data();
  load_card_value();
});

function load_card_value(){
  $("#num_day").html((parseFloat(data_day[data_day.length-1]).toFixed(0)).toString());
  $("#num_positive").html((parseFloat(data_num_positive[data_num_positive.length-1]).toFixed(0)).toString());
  $("#num_tested").html((parseFloat(data_num_tested[data_num_tested.length-1]).toFixed(0)).toString());
  $("#num_hospitalised").html((parseFloat(data_num_hospitalised[data_num_hospitalised.length-1]).toFixed(0)).toString());
  $("#num_cumulative_hospitalizations").html((parseFloat(data_num_cumulative_hospitalizations[data_num_cumulative_hospitalizations.length-1]).toFixed(0)).toString());
  $("#num_critical").html((parseFloat(data_num_critical[data_num_critical.length-1]).toFixed(0)).toString());
  $("#num_fatalities").html((parseFloat(data_num_fatalities[data_num_fatalities.length-1]).toFixed(0)).toString());




}
function load_card_kpi_value(){


  var percentChange_positive = data_num_positive[data_num_positive.length-1]-data_num_positive[data_num_positive.length-2]
  var percentChange_hos = data_num_hospitalised[data_num_hospitalised.length-1]-data_num_positive[data_num_hospitalised.length-2]
  var percentChange_cum_hos = data_num_hospitalised[data_num_cumulative_hospitalizations.length-1]-data_num_positive[data_num_cumulative_hospitalizations.length-2]
  var percentChange_crit = data_num_critical[data_num_critical.length-1]-data_num_critical[data_num_critical.length-2]
  var percentChange_fat = data_num_fatalities[data_num_fatalities.length-1]-data_num_fatalities[data_num_fatalities.length-2]

  if(percentChange_positive == 0){
    result = `
    <td class="fas fa-caret-right text-success">${ percentChange_positive }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#casesKpi").html(result)
  }
  else if(percentChange_positive > 0){
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_positive }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#casesKpi").html(result)
  }
  else{
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_positive }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#casesKpi").html(result)
  }


  if(percentChange_hos == 0){
    result = `
    <td class="fas fa-caret-right text-success">${ percentChange_hos }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_hospitalisedKpi").html(result)
  }
  else if(percentChange_hos < 0){
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_hos }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_hospitalisedKpi").html(result)
  }
  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_hos }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_hospitalisedKpi").html(result)
  }

    if(percentChange_cum_hos == 0){
      result = `
      <td class="fas fa-caret-right text-success">${ percentChange_cum_hos }</td>
      <td class="fas fa-percent text-success"></td>`
      $("#num_cmulative_hospitalizationsKpi").html(result)
    }
    else if(percentChange_cum_hos < 0){
      result = `
      <td class="fas fa-caret-down fa-1x text-success">${ percentChange_cum_hos }</td>
      <td class="fas fa-percent text-success"></td>`
      $("#num_cmulative_hospitalizationsKpi").html(result)
    }
    else{
      result = `
      <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_cum_hos }</td>
      <td class="fas fa-percent text-danger"></td>`
      $("#num_cmulative_hospitalizationsKpi").html(result)
    }

  if(percentChange_crit == 0){
    result = `
    <td class="fas fa-caret-right fa-1x text-success">${ percentChange_crit }</td>
    <td class="fas fa-percent text-success"></td>`

    $("#num_criticalKpi").html(result)
  }
  else if(percentChange_crit < 0){
    result = `


    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_crit }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_criticalKpi").html(result)
  }

  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_crit }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_criticalKpi").html(result)
  }
  if(percentChange_fat == 0){
    result = `
    <td class="fas fa-caret-right fa-1x text-success">${ percentChange_fat }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_fatalitiesKpi").html(result)
  }
  else if(percentChange_fat < 0){
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_fat }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_fatalitiesKpi").html(result)
  }

  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_fat  }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_fatalitiesKpi").html(result)
  }




}
function draw_current_data() {

  if (typeof(myLineChart_positive) != "undefined"){

    myLineChart_positive.destroy();

  }

  ctx_positive = document.getElementById("myAreaChartPositive");
  myLineChart_positive = new Chart(ctx_positive, {
    type: 'line',
    data: {
      labels: data_day,
      datasets: [
        // positives
        {
          label: "Positives ",
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
          data: positives_draw(),
        },
        // Hospitalised
        {
          label: "Hospitalised ",
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
          data: hospitalized_draw(),
        },
        // Hospitalised
        {
          label: "Hospitalised ",
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
          data: cum_hospitalized_draw(),
        },
        // Criticals
        {
          label: "criticals ",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.3)",
          borderColor: "rgba(255, 193, 7,1)",
          pointRadius: 4,
          pointBackgroundColor: "rgba(255, 193, 7,1)",
          pointBorderColor: "rgba(255, 193, 7,1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(255, 193, 7, 1)",
          pointHoverBorderColor: "rgba(255, 193, 7, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: criticals_draw(),
        },
        // Fatalies
        {
          label: "Fatalies ",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: fatalities_draw(),
        },






    ],
    },
    options: {

      annotation: {
        annotations: [
          {
            type: "line",
            mode: "vertical",
            scaleID: "x-axis-0",
            value: "16",
            borderColor: "red",
            label: {
              content: "TODAY",
              enabled: true,
              position: "top"
            }
          }
        ]
      },

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

function draw() {

  if (typeof(myLineChart) != "undefined"){

    myLineChart.destroy();

  }


  ctx_active_cases = document.getElementById("myAreaSeirModel").getContext("2d");;
  myLineChart = new Chart(ctx_active_cases, {
    type: 'line',

    data: {
      labels: data_day_seir,
      datasets: [
        // Susceptible
        {
          label: "Susceptible ",
          lineTension: 0.6,
          backgroundColor: "rgba(78, 115, 223, 0.2)",
          borderColor: "rgba(78, 115, 223, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(78, 115, 223, 1)",
          pointBorderColor: "rgba(78, 115, 223, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
          pointHoverBorderColor: "rgba(78, 115, 223, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: susceptible_draw(),
        },
        // Exposed
        {
          label: "Exposed ",
          lineTension: 0.6,
          backgroundColor: "rgba(240, 173, 78, 0.2)",
          borderColor: "rgba(240, 173, 78, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(240, 173, 78, 1)",
          pointBorderColor: "rgba(240, 173, 78, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(240, 173, 78, 1)",
          pointHoverBorderColor: "rgba(240, 173, 78, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: exposed_draw(),
        },
        // Infectious
        {
          label: "Infectious ",
          lineTension: 0.3,
          backgroundColor: "rgba(237, 0, 59,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: infected_draw(),
        },
        // Recovered
        {
          label: "Recovered ",
          lineTension: 0.6,
          backgroundColor: "rgba(37, 56, 60, 0.2)",
          borderColor: "rgba(37, 56, 60, 0.1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointHoverBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: recovered_draw(),
        },
        // Hospitalized
        {
          label: "hospitalized ",
          lineTension: 0.6,
          backgroundColor: "rgba(34,139,34, 0.2)",
          borderColor: "rgba(34,139,34, 0.1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(34,139,34, 0.1)",
          pointBorderColor: "rgba(34,139,34, 0.1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(34,139,34, 0.1)",
          pointHoverBorderColor: "rgba(34,139,34, 0.1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: hospitalized_seir_draw(),
        },

        // Criticals
        {
          label: "Criticals ",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.3)",
          borderColor: "rgba(255, 193, 7,1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(255, 193, 7,1)",
          pointBorderColor: "rgba(255, 193, 7,1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(255, 193, 7, 1)",
          pointHoverBorderColor: "rgba(255, 193, 7, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: criticals_seir_draw(),
        },
        // Fatalities
        {
          label: "Fatalities",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: fatalities_seir_draw(),
        },
        // Fatalities
        {
          label: "new Fatalities",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data:cum_bed2_draw(),
        },
        // Recovered
        {
          label: "Bed available",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0, 0)",
          borderColor: "rgba(0, 0, 0, 0)",
          pointRadius: 2,
          pointBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointHoverBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: cum_bed_draw(),
        },
        // Recovered
        {
          label: "Bed available in ICU",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0, 0)",
          borderColor: "rgba(0, 0, 0, 0)",
          pointRadius: 2,
          pointBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointHoverBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: cum_icu_draw(),
        },

        // Hospitalised
        {
          label: "Hospitalised ",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0)",
          borderColor: "rgba(0, 0, 0,0)",
          pointRadius: 4,
          pointBackgroundColor: "rgba(34,139,34, 0.1)",
          pointBorderColor: "rgba(34,139,34, 0.1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(34,139,34, 0.1)",
          pointHoverBorderColor: "rgba(34,139,34, 0.1)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: hospitalized_draw(),
        },


        // Criticals
        {
          label: "criticals ",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0)",
          borderColor: "rgba(0, 0, 0,0)",
          pointRadius: 4,
          pointBackgroundColor: "rgba(255, 193, 7,1)",
          pointBorderColor: "rgba(255, 193, 7,1)",
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "rgba(255, 193, 7, 1)",
          pointHoverBorderColor: "rgba(255, 193, 7, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: criticals_draw(),
        },
        // Fatalies
        {
          label: "Fatalies ",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0)",
          borderColor: "rgba(0, 0, 0,0)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 4,
          data: fatalities_draw(),
        },
        {
          label: "new Fatalies ",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0)",
          borderColor: "rgba(0, 0, 0,0)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: cum_icu2_draw(),
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
