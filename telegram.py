from utils.function import load_data, train_svi_with_stats, plot_training_results_with_stats, send_telegram_message



if __name__ == "__main__":
    #send_telegram_message(
    #title="Training Completed",
    #message=f"Hello"
    #)

    accuracy = 0.95643096
    experiment_serial = "20231001-123456"  # Example serial, replace with actual value

    send_telegram_message(
    title="Training Completed",
    message=f"Confusion Matrix Accuracy: {accuracy * 100:.6f}%\n"
            f"Results plot saved as: results_eurosat/bayesian_cnn_training_results_{experiment_serial}.png"
)