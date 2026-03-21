nextflow.enable.dsl=2

params.training_dataset = null
params.test_dataset = null
params.outdir = "results"
params.output_name = "Actin_Myosin_Predictions.csv"
params.pf00063_hmm = null
params.hmm_evalue = 1e-5
params.min_fragment_length = 500
params.hard_negative_score_cutoff = 4
params.test_fasta = null

if( !params.training_dataset ) {
    error "Please specify --training_dataset"
}

if( !params.test_dataset ) {
    error "Please specify --test_dataset"
}

if( !params.pf00063_hmm ) {
    error "Please specify --pf00063_hmm"
}

if( !params.test_fasta ) {
    error "Please specify --test_fasta"
}

process TRAIN_AND_PREDICT_ACTIN_MYOSIN {
    tag "Actin-Myosin_Prediction"

    publishDir "${params.outdir}/predictions_results", mode: 'copy'

    input:
    path training_dataset
    path test_dataset

    output:
    path "prediction_outputs/${params.output_name}", emit: prediction_csv
    path "prediction_outputs/*"

    script:
    """
    mkdir -p prediction_outputs

    python3 ${projectDir}/bin/Train_Test_Actin_Myosin.py \
        --training_dataset ${training_dataset} \
        --test_dataset ${test_dataset} \
        --output_dir prediction_outputs \
        --output_name ${params.output_name}
    """
}

process CROSS_VALIDATION {
    tag "Actin-Myosin_CrossValidation"
    publishDir "${params.outdir}/cross_validation_results", mode: 'copy'

    input:
    path training_dataset

    output:
    path "crossval_outputs/*"

    script:
    """
    mkdir -p crossval_outputs

    python3 ${projectDir}/bin/Cross_Validation_Actin_Myosin.py \
        --training_dataset ${training_dataset} \
        --output_dir crossval_outputs
    """
}

process FILTER_CONFIRMED_PREDICTIONS {
    tag "Filter_Confirmed_Predictions"

    publishDir "${params.outdir}/confirmed_predictions", mode: 'copy'

    input:
    path prediction_csv
    path test_fasta

    output:
    path "confirmed_outputs/Predicted_Actins.csv", emit: actin_csv
    path "confirmed_outputs/Predicted_Myosins.csv", emit: myosin_csv
    path "confirmed_outputs/Predicted_Myosins.fasta", emit: myosin_fasta

    script:
    """
    mkdir -p confirmed_outputs

    python3 ${projectDir}/bin/GetPredictedSequences.py \
        --input_csv ${prediction_csv} \
        --input_fasta ${test_fasta} \
        --output_actin_csv confirmed_outputs/Predicted_Actins.csv \
        --output_myosin_csv confirmed_outputs/Predicted_Myosins.csv \
        --output_myosin_fasta confirmed_outputs/Predicted_Myosins.fasta
    """
}

process RUN_HMMSEARCH_PF00063 {
    tag "HMMSEARCH_PF00063"

    publishDir "${params.outdir}/pfam_validation", mode: 'copy'

    input:
    path myosin_fasta
    path pf00063_hmm

    output:
    path "pfam_outputs/pf00063_hits.tbl", emit: tblout
    path "pfam_outputs/pf00063_domtblout.tbl", emit: domtblout
    path "pfam_outputs/*"

    script:
    """
    mkdir -p pfam_outputs

    hmmsearch \
        --tblout pfam_outputs/pf00063_hits.tbl \
        --domtblout pfam_outputs/pf00063_domtblout.tbl \
        -E ${params.hmm_evalue} \
        ${pf00063_hmm} \
        ${myosin_fasta} > pfam_outputs/pf00063_hmmsearch_stdout.txt
    """
}

process BUILD_MYOSIN_FEEDBACK {
    tag "Build_Myosin_Feedback"

    publishDir "${params.outdir}/feedback_builder", mode: 'copy'

    input:
    path prediction_csv
    path myosin_csv
    path domtblout

    output:
    path "feedback_outputs/*"

    script:
    """
    mkdir -p feedback_outputs

    python3 ${projectDir}/bin/Build_Myosin_Feedback.py \
        --prediction_csv ${prediction_csv} \
        --myosin_csv ${myosin_csv} \
        --hmmer_domtblout ${domtblout} \
        --output_feedback_csv feedback_outputs/Myosin_Feedback_Table.csv \
        --output_confirmed_csv feedback_outputs/Domain_Confirmed_Myosins.csv \
        --output_hard_negative_csv feedback_outputs/Hard_Negative_Candidates.csv \
        --output_uncertain_csv feedback_outputs/Uncertain_Fragments.csv \
        --min_fragment_length ${params.min_fragment_length} \
        --hard_negative_score_cutoff ${params.hard_negative_score_cutoff}
    """
}

workflow {
    training_ch = Channel.fromPath(params.training_dataset, checkIfExists: true)
    test_ch = Channel.fromPath(params.test_dataset, checkIfExists: true)
    test_fasta_ch = Channel.fromPath(params.test_fasta, checkIfExists: true)
    pf00063_hmm_ch = Channel.fromPath(params.pf00063_hmm, checkIfExists: true)

    predictions = TRAIN_AND_PREDICT_ACTIN_MYOSIN(training_ch, test_ch)
    CROSS_VALIDATION(training_ch)
    
    confirmed = FILTER_CONFIRMED_PREDICTIONS(predictions.prediction_csv, test_fasta_ch)

    pfam_results = RUN_HMMSEARCH_PF00063(confirmed.myosin_fasta, pf00063_hmm_ch)

    BUILD_MYOSIN_FEEDBACK(predictions.prediction_csv, confirmed.myosin_csv, pfam_results.domtblout)
}
