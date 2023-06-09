import logging
import os

from s3_helper import S3Ref, copy_with_query, create_ref_at_parent_key,query_helper_pretrain

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    This method selects 10% of the input manifest as validation and creates an s3 file containing the validation objects.
    """
    label_attribute_name = event["LabelAttributeName"]
    meta_data = event["meta_data"]
    s3_input_uri = meta_data["IntermediateManifestS3Uri"]
    pretrain_model=event["meta_data"]["PretrainedModel"]

    input_total = int(meta_data["counts"]["input_total"])
    # 10% of the total input should be used for validation.
    validation_set_size = input_total // 10

    source = S3Ref.from_uri(s3_input_uri)

    validation_labeled_query = """select * from s3object[*] s where s."{}-metadata"."human-annotated" IN ('yes') LIMIT {}""".format(
        label_attribute_name, validation_set_size
    )
    dest = create_ref_at_parent_key(source, "validation_input.manifest")
    pretrain_validation_manifest = dest
    copy_with_query(source, dest, validation_labeled_query)
    logger.info(
        "Uploaded validation set of size {} to {}.".format(validation_set_size, dest.get_uri())
    )
    #Create validation  dataset for pretrained algorithm
    
    if pretrain_model == 'true' :
        source=dest
        #dest = S3Ref.from_uri(source +'pretrain-batch-transform-validation/' +  "data.jsonl")
        dest=create_ref_at_parent_key(source, "pretrain-batch-transform-validation/data.csv")
        validation_labeled_query = """select s."category" , s."source"  from s3object[*] s where s."{}-metadata"."human-annotated" IN ('yes') LIMIT {}""".format(
            label_attribute_name,validation_set_size)
        query_helper_pretrain(source,dest,validation_labeled_query,task="validation")
        
        logger.info("Uploaded modified validation data for inference to {}.".format(dest.get_uri()))

    meta_data["counts"]["validation"] = validation_set_size
    meta_data["ValidationS3Uri"] = dest.get_uri()
    meta_data["Pretrain_validationS3Uri_Manifest"]=pretrain_validation_manifest.get_uri()
    return meta_data
