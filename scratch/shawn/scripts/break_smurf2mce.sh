echo '-> Setting num_averages to 100'
./smurf2mce_must_die.sh 100
echo '-> Waiting 10 seconds'
sleep 10
echo '-> Setting num_averages to 10'
./smurf2mce_must_die.sh 10
