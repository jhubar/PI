$(document).ready(function() {

  const $value_spreading_period = $('.value_spreading_period');
  const $value = $('#range_spreading_period');
  $value_spreading_period.html($value.val());
  $value.on('input change', () => {

    $value_spreading_period.html($value.val());
  });
});
